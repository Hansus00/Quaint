from numpy.typing import NDArray
import numpy as np


class Potential:
    matrix: NDArray[np.complex128]

    def __init__(self, matrix: NDArray[np.complex128]):
        raise NotImplementedError
