from numpy.typing import NDArray
import numpy as np


class Potential:
    matrix: NDArray[np.float128]

    def __init__(self, matrix: NDArray[np.float128]):
        self.matrix = matrix


class InfiniteWellPotential(Potential):
    def __init__(self, size_x: int, size_y: int, WALL_VALUE: float = 100):
        matrix = np.zeros((size_x, size_y), dtype=np.float128)

        for x in range(0, size_x):
            matrix[x, 0] = WALL_VALUE
            matrix[x, size_y - 1] = WALL_VALUE

        for y in range(0, size_y):
            matrix[0, y] = WALL_VALUE
            matrix[size_x - 1, y] = WALL_VALUE

        super().__init__(matrix)
