from numpy.typing import NDArray
import numpy as np


class StationaryWaveFunc:
    matrix: NDArray[np.complex128]

    def __init__(self, matrix: NDArray[np.complex128]):
        self.matrix = matrix


class GaussianPacket(StationaryWaveFunc):
    def __init__(
        self,
        r0: tuple[int, int],
        k0: NDArray[np.float64],
        sigma0: NDArray[np.float64],
        size_x: int,
        size_y: int,
    ):
        # TODO: fix it!
        _x = np.linspace(0, size_x, size_x - 1)
        _y = np.linspace(0, size_y, size_y - 1)
        x,y = np.meshgrid(_x, _y)

        dx = x - r0[0]
        dy = y - r0[1]

        dr = np.stack([dx, dy])

        matrix = np.exp(1j * np.einsum("i,ijk->jk",k0,dr) - 0.5 * np.einsum("ikl,ij,jkl->kl",dr,np.linalg.inv(sigma0),dr))
        matrix /= np.sum(np.abs(matrix)**2)
        super().__init__(matrix)
        
