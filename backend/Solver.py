from StationaryWaveFunc import StationaryWaveFunc
from Potential import Potential
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from numpy.typing import NDArray
import numpy as np


class Solver:
    potential: Potential
    delta_t: float
    L_2D: NDArray[np.complex128]
    H: NDArray[np.complex128]

    def __init__(self, potential: Potential, delta_t: float = 1e-3):
        self.potential = potential
        self.delta_t = delta_t

        # initializing laplace operator
        Nx, Ny = self.potential.matrix.shape

        dx, dy = 1, 1
        D_xx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
        D_yy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2

        # self.L_2D = sp.kron(D_xx, I_y) + sp.kron(I_x, D_yy) does the same
        I_x = sp.identity(Nx)
        I_y = sp.identity(Ny)

        self.L_2D = sp.kron(I_y, D_xx) + sp.kron(D_yy, I_x)

    def __call__(
        self,
        wave_func: StationaryWaveFunc,
        n_steps: int = 1,
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class CrankNicolson(Solver):
    def __init__(self, potential: Potential, delta_t: float = 1e-3):
        super().__init__(potential, delta_t)

    def __call__(
        self,
        wave_func: StationaryWaveFunc,
        n_steps: int = 1,
    ) -> StationaryWaveFunc:
        Nx, Ny = self.potential.matrix.shape

        # calculating Hamilton operator
        T_matrix = -(1 / (2 * wave_func.mass)) * self.L_2D
        V_1d = self.potential.matrix.flatten()
        V_matrix = sp.diags(V_1d, offsets=0, format="csr")
        self.H = T_matrix + V_matrix

        # Cayley method
        I = sp.eye(Nx * Ny, format="csr")
        prefactor = (1j * self.delta_t) / 2
        A = (I + prefactor * self.H).astype(np.complex128)
        B = (I - prefactor * self.H).astype(np.complex128)

        psi_1d = wave_func.matrix.flatten().astype(np.complex128)
        psi_1d_new = spsolve(A, B.dot(psi_1d))

        psi_2d_new = psi_1d_new.reshape((Nx, Ny))

        return StationaryWaveFunc(np.array(psi_2d_new), wave_func.mass)


class Constant(Solver):
    def __init__(self, potential: Potential):
        super().__init__(potential)

    def __call__(
        self, wave_func: StationaryWaveFunc, n_steps: int = 1
    ) -> StationaryWaveFunc:
        return wave_func


class SSFM(Solver):
    def __init__(self):
        raise NotImplementedError


if __name__ == "__main__":
    # TODO: usunąć
    from Potential import InfiniteWellPotential
    from StationaryWaveFunc import GaussianPacket

    ipw = InfiniteWellPotential(3, 3, 1e5)
    cn = CrankNicolson(ipw)
    print(np.array(cn.L_2D))
    wf = GaussianPacket(
        (10, 10), np.array([1, 1]), np.array([[1, 0], [0, 1]]), 1, *ipw.matrix.shape
    )

    wf = cn(wf)
    # print(wf.matrix)
