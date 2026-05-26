from typing import Callable, cast
import logging
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.fft import dstn, idstn

from .Potential import Potential
from .StationaryWaveFunc import StationaryWaveFunc

logger = logging.getLogger(__name__)

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
        logger.info("---------------New simulation---------------")
        logger.info(
            "Physical size of the simulation (L_x,L_y): %s, %s",
            self.potential.matrix.shape[0] * self._dx,
            self.potential.matrix.shape[1] * self._dy,
        )
        if grid_step > 1:
            logger.warning("Grid step might be too big")

        self.setup()
        self._stability_conditions()

    def setup(self):
        """Override in subclasses to initialize numerical matrices/propagators
        before stability conditions are checked."""
        raise NotImplementedError

    def _stability_conditions(self):
        """Check whether Courant-Friedrichs-Lewy and Nyquist conditions
        are satisfied. Requires ev_energy() to work."""
        assert SAFETY_FACTOR <= 1

        ev_energy = self.ev_energy()

        logger.info("<E> = %s", ev_energy)
        k = np.sqrt(2 * self._wave_func.mass * np.abs(ev_energy))
        k_max = np.pi / (np.mean([self._dx, self._dy]))  # Nyquist

        logger.info(r"k_{max} = %s", k_max)
        logger.info(r"|k_0| = %s", np.abs(k))
        if np.abs(k) >= k_max * SAFETY_FACTOR:
            logger.warning(
                "Nyquist condition (|k_0| < %s * k_{max}) is not satisfied, "
                "decrease grid_step",
                SAFETY_FACTOR,
            )
        courant_number = k / self._wave_func.mass * self.delta_t / self._dx
        logger.info(r"C = %s", courant_number)
        if courant_number > SAFETY_FACTOR:
            logger.warning(
                "Courant number should be << 1, but is %s, "
                "decrease delta_t or increase grid_step",
                courant_number,
            )

    def step(self) -> None:
        """Evolves one step of wave function after t + Delta t"""
        self._steps_evolved += 1

    def get_steps_evolved(self) -> int:
        """Returns number of evolved steps"""
        return self._steps_evolved

    def update(self, n_step: int = 1) -> StationaryWaveFunc:
        """Returns evolved n steps of wave function after t + n * dt"""
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

    def setup(self):
        Nx, Ny = self.potential.matrix.shape

        self.L_2D = self._create_laplace_operator(Nx, Ny)
        self.H = self._create_hamilton_operator(self.L_2D, self._wave_func.mass)
        self.A, self.B = self._create_cayley_matrices(Nx * Ny, self.H)
        self._factorized_A = factorized(
            sp.csc_matrix(self.A)
        )  # factorize once for the whole simulation

        self._wave_state_1D = self._wave_func.matrix.flatten().astype(np.complex128)

    def _create_laplace_operator(self, Nx: int, Ny: int) -> sp.spmatrix:
        # TODO: add periodic boundary conditions
        # similar to https://stackoverflow.com/questions/34895970
        D_xx = (
            sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx), dtype=np.float64)
            / self._dx**2
        )  # type: ignore
        D_yy = (
            sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny), dtype=np.float64)
            / self._dy**2
        )  # type: ignore

        I_x = sp.identity(Nx)
        I_y = sp.identity(Ny)

        # 2D Laplacian discretization consistent with C-order flattening
        # used by numpy:
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
        ident = sp.eye(N_total, format="csr")
        prefactor = (1j * self.delta_t) / 2
        A = (ident + prefactor * H).astype(np.complex128)
        B = (ident - prefactor * H).astype(np.complex128)
        return A, B

    def step(self):
        super().step()
        # equivalent to spsolve(self.A, self.B @ self._wave_state_1D)
        # but with pre-factorized matrix A
        _tmp = self.B.dot(self._wave_state_1D)
        self._wave_state_1D = self._factorized_A(_tmp)  # type: ignore

    def get_wave_function(self) -> StationaryWaveFunc:
        Nx, Ny = self.potential.matrix.shape

        self._wave_func = StationaryWaveFunc(
            np.array(self._wave_state_1D.reshape((Nx, Ny))), self._wave_func.mass
        )

        return self._wave_func

    def ev_energy(self) -> np.complex128:
        """Returns expected value of the hamiltonian.
        There may be some cases where H is not hermitian."""
        denom = np.sum(np.conjugate(self._wave_state_1D) * self._wave_state_1D)
        return (
            np.sum(
                np.conjugate(self._wave_state_1D)
                * (self.H @ self._wave_state_1D)  # type: ignore
            )
            / denom
        )


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

    def setup(self):
        Nx, Ny = self.potential.matrix.shape

        self._U_V = self._create_real_space_propagator()
        self._U_T = self._create_momentum_propagator(Nx, Ny, self._wave_func.mass)

    def _create_real_space_propagator(self) -> NDArray[np.complex128]:
        raise NotImplementedError

    def _create_momentum_propagator(
        self, Nx: int, Ny: int, mass: float
    ) -> NDArray[np.complex128]:
        """Creates the momentum space propagator."""
        kx = np.arange(1, Nx + 1) * np.pi / ((Nx + 1) * self._dx)
        ky = np.arange(1, Ny + 1) * np.pi / ((Ny + 1) * self._dy)
        kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")

        T = (kx2 + ky2) / (2 * mass)
        return np.exp(-1j * T * self.delta_t)

    def ev_energy(self) -> np.complex128:
        psi = self._wave_func.matrix
        Nx, Ny = psi.shape

        psi_k = dstn(psi, type=1)

        kx = np.arange(1, Nx + 1) * np.pi / ((Nx + 1) * self._dx)
        ky = np.arange(1, Ny + 1) * np.pi / ((Ny + 1) * self._dy)
        kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")

        # kinetic energy operator in k-space
        T = (kx2 + ky2) / (2 * self._wave_func.mass)

        # expected value of kinetic energy calculated in sine-basis
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

        psi_k = cast(NDArray[np.complex128], dstn(psi, type=1))
        psi_k *= self._U_T
        psi = cast(NDArray[np.complex128], idstn(psi_k, type=1))

        prob = np.sqrt(np.sum(np.abs(psi) ** 2))
        self._wave_func = StationaryWaveFunc(psi / prob, self._wave_func.mass)


class SSFMSymmetric(_BaseSSFM):
    """Symmetric Split-Step Fourier Method. Accurate to O(delta_t**3)."""

    def _create_real_space_propagator(self) -> NDArray[np.complex128]:
        """Creates the half-step real space propagator."""
        return np.exp(-1j * self.potential.matrix * self.delta_t / 2)

    def step(self):
        super().step()

        psi = self._wave_func.matrix * self._U_V

        psi_k = cast(NDArray[np.complex128], dstn(psi, type=1))
        psi_k *= self._U_T
        psi = cast(NDArray[np.complex128], idstn(psi_k, type=1))

        psi *= self._U_V

        prob = np.sqrt(np.sum(np.abs(psi) ** 2))
        self._wave_func = StationaryWaveFunc(psi / prob, self._wave_func.mass)
