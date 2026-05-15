from numpy.typing import NDArray
import numpy as np


class StationaryWaveFunc:
    matrix: NDArray[np.complex128]

    def __init__(self, matrix: NDArray[np.complex128]):
        raise NotImplementedError


class GaussianPackage(StationaryWaveFunc):
    def __init__(
        self,
        r0: tuple[float, float],
        k0: tuple[float, float],
        sigma0: tuple[float, float],
    ):
        # TODO: fix it!
        matrix = np.eye(10) + np.eye(10) * 1.0j
        super().__init__(matrix)
