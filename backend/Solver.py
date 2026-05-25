from typing import Callable
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from .Potential import Potential
from .StationaryWaveFunc import StationaryWaveFunc

RED = "\033[91m"  # for text coloring
RESET = "\033[0m"  # for text coloring
SAFETY_FACTOR = 0.3
"""Conservative scaling applied to theoretical limits.

For a maximum allowed value x_max, the effective operational limit becomes:
    x_safe = SAFETY_FACTOR * x_max
"""


class _Solver:
    potential: Potential
    delta_t: float
    _wave_func: StationaryWaveFunc
    _steps_evolved: int = 0
    _dx: float = 1
    _dy: float = 1

    def __init__(
        self,
        potential: Potential,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        grid_step: float = 1,
    ):
        assert potential.matrix.shape == wave_func.matrix.shape

        self.potential = potential
        self._wave_func = wave_func
        self.delta_t = delta_t
        self._dx, self._dy = grid_step, grid_step
        print(
            "\n\n\n---------------New simulation---------------",
            "\nPhysical size of the simulation (L_x,L_y):",
            self.potential.matrix.shape[0] * self._dx,
            self.potential.matrix.shape[1] * self._dy,
            "\n",
        )
        if grid_step > 1:
            warnings.warn(RED + "Grid step might be too big" + RESET)

    def _stability_conditions(self):
        """Check wether Courant–Friedrichs–Lewy and Nyquist conditions are satisfied
        Requires ev_energy() to work, shold be the last function run in __init__"""
        assert SAFETY_FACTOR <= 1

        ev_energy = self.ev_energy()
        print("<E> =", ev_energy)
        k = np.sqrt(2 * self._wave_func.mass * np.abs(ev_energy))
        k_max = np.pi / (np.mean([self._dx, self._dy]))  # Nyquist

        print(r"k_{max} =", k_max)
        print(r"|k_0| =", np.abs(k))
        if np.abs(k) >= k_max * SAFETY_FACTOR:
            warnings.warn(
                RED
                + "Nyquist condition (|k_0| < "
                + str(SAFETY_FACTOR)
                + r"k_{max}) is not satisfied, decrease grid_step"
                + RESET,
                RuntimeWarning,
            )
        courant_number = k / self._wave_func.mass * self.delta_t / self._dx
        print(r"C =", courant_number)
        if courant_number > SAFETY_FACTOR:
            warnings.warn(
                RED
                + "Courant number should be << 1, but is "
                + str(courant_number)
                + ", decrease delta_t or increase grid_step"
                + RESET,
                RuntimeWarning,
            )
        print("\n")

    def step(self) -> None:
        """Evolves on step of wave function after t + Delta t"""
        self._steps_evolved += 1

    def get_steps_evolved(self) -> int:
        """Returns number ov evolves steps"""
        return self._steps_evolved

    def update(self, n_step: int = 1) -> StationaryWaveFunc:
        """Returns evolved n steps of wave function after time t + n * Delta t"""
        for i in range(0, n_step):
            self.step()

        return self.get_wave_function()

    def get_wave_function(self) -> StationaryWaveFunc:
        """Returns wave function at current state at time t"""
        return self._wave_func

    def ev_energy(self) -> np.complex128:
        raise NotImplementedError


class CrankNicolson(_Solver):
    L_2D: sp.spmatrix
    H: sp.spmatrix
    A: sp.spmatrix | sp.sparray
    B: sp.spmatrix | sp.sparray
    _factorized_A: Callable

    _wave_state_1D: NDArray[np.complex128]

    def __init__(
        self,
        potential: Potential,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        grid_step: float = 1,
    ):
        super().__init__(potential, wave_func, delta_t, grid_step)

        Nx, Ny = self.potential.matrix.shape

        self.L_2D = self._create_laplace_operator(Nx, Ny)
        self.H = self._create_hamilton_operator(self.L_2D, wave_func.mass)
        self.A, self.B = self._create_cayley_matrices(Nx * Ny, self.H)
        self._factorized_A = factorized(
            sp.csc_matrix(self.A)
        )  # factorize once for the whole simulation

        self._wave_state_1D = self._wave_func.matrix.flatten().astype(np.complex128)
        self._stability_conditions()

    def _create_laplace_operator(self, Nx: int, Ny: int) -> sp.spmatrix:
        # TODO: add periodic boundary conditions
        # similiar to https://stackoverflow.com/questions/34895970/buildin-a-sparse-2d-laplacian-matrix-using-scipy-modules
        D_xx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx), dtype=np.float64) / self._dx**2  # type: ignore
        D_yy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny), dtype=np.float64) / self._dy**2  # type: ignore

        I_x = sp.identity(Nx)
        I_y = sp.identity(Ny)

        # 2D Laplacian discretization consistent with C-order flattening used by numpy:
        # index mapping (i, j) -> i * Ny + j
        return sp.kron(D_xx, I_y) + sp.kron(I_x, D_yy)

    def _create_hamilton_operator(self, L_2D: sp.spmatrix, mass: float) -> sp.spmatrix:
        T_matrix = -(1 / (2 * mass)) * L_2D
        V_1d = self.potential.matrix.flatten()
        V_matrix = sp.diags(V_1d, offsets=0, format="csr")
        return T_matrix + V_matrix

    def _create_cayley_matrices(
        self, N_total: int, H: sp.spmatrix
    ) -> tuple[sp.spmatrix | sp.sparray, sp.spmatrix | sp.sparray]:
        I = sp.eye(N_total, format="csr")
        prefactor = (1j * self.delta_t) / 2
        A = (I + prefactor * H).astype(np.complex128)
        B = (I - prefactor * H).astype(np.complex128)
        return A, B

    def step(self):
        super().step()
        # does the same as np.asarray(spsolve(self.A, self.B @ self._wave_state_1D)), but with pre-factorized matrix A
        self._wave_state_1D = self._factorized_A(self.B.dot(self._wave_state_1D))  # type: ignore

    def get_wave_function(self) -> StationaryWaveFunc:
        Nx, Ny = self.potential.matrix.shape

        self._wave_func = StationaryWaveFunc(
            np.array(self._wave_state_1D.reshape((Nx, Ny))), self._wave_func.mass
        )

        return self._wave_func

    def ev_energy(self) -> np.complex128:
        """Returns expected value of the hamiltonian.
        There may be some cases where H is not hermitian."""
        return np.sum(
            np.conjugate(self._wave_state_1D) * (self.H @ self._wave_state_1D)  # type: ignore
        ) / np.sum(np.conjugate(self._wave_state_1D) * self._wave_state_1D)


class Constant(_Solver):
    pass


class _BaseSSFM(_Solver):
    """Base class for Split-Step Fourier Methods."""

    _U_T: NDArray[np.complex128]
    _U_V: NDArray[np.complex128]

    def __init__(
        self,
        potential: Potential,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        grid_step: float = 1,
    ):
        super().__init__(potential, wave_func, delta_t, grid_step)

        Nx, Ny = self.potential.matrix.shape

        self._U_V = self._create_real_space_propagator()
        self._U_T = self._create_momentum_propagator(Nx, Ny, wave_func.mass)
        self._stability_conditions()

    def _create_real_space_propagator(self) -> NDArray[np.complex128]:
        raise NotImplementedError

    def _create_momentum_propagator(
        self, Nx: int, Ny: int, mass: float
    ) -> NDArray[np.complex128]:
        """Creates the momentum space propagator."""
        kx = np.fft.fftfreq(Nx, d=self._dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=self._dy) * 2 * np.pi
        kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")

        T = (kx2 + ky2) / (2 * mass)
        return np.exp(-1j * T * self.delta_t)

    def ev_energy(self) -> np.complex128:
        psi = self._wave_func.matrix
        psi_k = np.fft.fft2(psi)

        kx = np.fft.fftfreq(self._wave_func.matrix.shape[0], d=self._dx) * 2 * np.pi
        ky = np.fft.fftfreq(self._wave_func.matrix.shape[1], d=self._dy) * 2 * np.pi
        kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")
        # kinetic energy in terms of k
        T = (kx2 + ky2) / (2 * self._wave_func.mass)

        # expected value of kinetic energy calculated in k-space
        ev_T = np.sum(np.conjugate(psi_k) * T * psi_k) / np.sum(
            np.conjugate(psi_k) * psi_k
        )

        # expected value of potential enrgy calculated in x-space
        ev_V = np.sum(np.conjugate(psi) * self.potential.matrix * psi) / np.sum(
            np.conjugate(psi) * psi
        )

        return ev_T + ev_V


class SSFM(_BaseSSFM):
    """Standard Split-Step Fourier Method. Accurate to O(delta_t**2)."""

    def _create_real_space_propagator(self) -> NDArray[np.complex128]:
        """Creates the full-step real space propagator."""
        return np.exp(-1j * self.potential.matrix * self.delta_t)

    def step(self):
        super().step()

        psi = self._wave_func.matrix * self._U_V

        psi_k = np.fft.fft2(psi)
        psi_k *= self._U_T
        psi = np.fft.ifft2(psi_k)

        self._wave_func = StationaryWaveFunc(psi, self._wave_func.mass)


class SSFMSymmetric(_BaseSSFM):
    """Symmetric Split-Step Fourier Method. Accurate to O(delta_t**3)."""

    def _create_real_space_propagator(self) -> NDArray[np.complex128]:
        """Creates the half-step real space propagator."""
        return np.exp(-1j * self.potential.matrix * self.delta_t / 2)

    def step(self):
        super().step()

        psi = self._wave_func.matrix * self._U_V

        psi_k = np.fft.fft2(psi)
        psi_k *= self._U_T
        psi = np.fft.ifft2(psi_k)

        psi *= self._U_V

        self._wave_func = StationaryWaveFunc(psi, self._wave_func.mass)
