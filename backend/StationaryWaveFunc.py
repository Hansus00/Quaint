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
        r = np.meshgrid(_x, _y)
        matrix = np.exp(1j * np.einsum("iijk->jk",k0,r) - 0.5 (r-r0).T @ np.linalg.inv(sigma0) @ (r-r0))
        matrix /= np.sum(np.abs(matrix)**2)
        super().__init__(matrix)
        
