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
        _x = np.arange(size_x)
        _y = np.arange(size_y)
        
        x, y = np.meshgrid(_x, _y, indexing="ij")

        dx = x - r0[0]
        dy = y - r0[1]

        dr = np.stack([dx, dy])

        matrix = np.exp(
            1j * np.einsum("i,ijk->jk", k0, dr) 
            - 0.5 * np.einsum("ikl,ij,jkl->kl", dr, np.linalg.inv(sigma0), dr)
        )
        
        matrix /= np.sqrt(np.sum(np.abs(matrix)**2))
        super().__init__(matrix)        
