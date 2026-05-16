from .StationaryWaveFunc import StationaryWaveFunc
from .Potential import Potential
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from numpy.typing import NDArray
import numpy as np


class Solver:
    potential: Potential
    delta_t: float
    _wave_func: StationaryWaveFunc

    def __init__(
        self, potential: Potential, wave_func: StationaryWaveFunc, delta_t: float = 1e-3
    ):
        self.potential = potential
        self._wave_func = wave_func
        self.delta_t = delta_t

    def step(self) -> None:
        """Evolves on step of wave function after t + Delta t"""
        raise NotImplementedError

    def update(self, n_step: int = 1) -> StationaryWaveFunc:
        """Returns evolved n steps of wave function after time t + n * Delta t"""
        for i in range(0, n_step):
            self.step()

        return self.get_wave_function()

    def get_wave_function(self) -> StationaryWaveFunc:
        """Returns wave function at current state at time t"""
        return self._wave_func


class CrankNicolson(Solver):
    L_2D: NDArray[np.complex128]  # TODO: to be removed
    H: NDArray[np.complex128]  # TODO: to be removed
    A: NDArray[np.complex128]
    B: NDArray[np.complex128]

    _wave_state_1D: NDArray[np.complex128]

    def __init__(
        self, potential: Potential, wave_func: StationaryWaveFunc, delta_t: float = 1e-3
    ):
        super().__init__(potential, wave_func, delta_t)

        # initializing laplace operator
        Nx, Ny = self.potential.matrix.shape

        dx, dy = 1, 1
        D_xx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
        D_yy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2

        # self.L_2D = sp.kron(D_xx, I_y) + sp.kron(I_x, D_yy) does the same
        I_x = sp.identity(Nx)
        I_y = sp.identity(Ny)

        self.L_2D = sp.kron(I_y, D_xx) + sp.kron(D_yy, I_x)

        # calculating Hamilton operator
        T_matrix = -(1 / (2 * wave_func.mass)) * self.L_2D
        V_1d = self.potential.matrix.flatten()
        V_matrix = sp.diags(V_1d, offsets=0, format="csr")
        self.H = T_matrix + V_matrix

        # Cayley method
        I = sp.eye(Nx * Ny, format="csr")
        prefactor = (1j * self.delta_t) / 2
        self.A = (I + prefactor * self.H).astype(np.complex128)
        self.B = (I - prefactor * self.H).astype(np.complex128)

        self._wave_state_1D = self._wave_func.matrix.flatten().astype(np.complex128)

    def step(self):
        # Crank Nicolson
        self._wave_state_1D = spsolve(self.A, self.B.dot(self._wave_state_1D))

    def get_wave_function(self) -> StationaryWaveFunc:
        Nx, Ny = self.potential.matrix.shape

        self._wave_func = StationaryWaveFunc(
            np.array(self._wave_state_1D.reshape((Nx, Ny))), self._wave_func.mass
        )

        return self._wave_func


class Constant(Solver):
    def __init__(
        self, potential: Potential, wave_func: StationaryWaveFunc, delta_t: float = 1e-3
    ):
        super().__init__(potential, wave_func, delta_t)

    def step(self):
        pass


class SSFM(Solver):
    def __init__(self):
        raise NotImplementedError
