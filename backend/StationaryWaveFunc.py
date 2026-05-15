from numpy.typing import NDArray
import numpy as np


class StationaryWaveFunc:
    matrix: NDArray[np.complex128]

    def __init__(self, matrix: NDArray[np.complex128]):
        self.matrix = matrix


class GaussianPacket(StationaryWaveFunc):
    def __init__(
        self,
        r0: tuple[float, float],
        k0: tuple[float, float],
        sigma0: NDArray[np.float64],
        size_x: float,
        size_y: float,
    ):
        # TODO: fix it!
        matrix = np.zeros((size_x, size_y)) * (1 + 1.0j)
        super().__init__(matrix)
