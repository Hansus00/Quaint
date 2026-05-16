from numpy.typing import NDArray
import numpy as np


class Potential:
    matrix: NDArray[np.float128]

    def __init__(self, matrix: NDArray[np.float128]):
        self.matrix = matrix

    def __add__(self, other: "Potential") -> "Potential":
        return Potential(self.matrix + other.matrix)


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


class GaussianBumpPotential(Potential):
    def __init__(
        self,
        size_x: int,
        size_y: int,
        r0: tuple[int, int],
        V0: float,
        sigma0: NDArray[np.float64],
    ):
        """Gaussian potential with peak at r0, peak value V0, and covariance matrix sigma0."""
        _x = np.arange(size_x)
        _y = np.arange(size_y)

        x, y = np.meshgrid(_x, _y, indexing="ij")

        dx = x - r0[0]
        dy = y - r0[1]

        dr = np.stack([dx, dy])

        matrix = V0 * np.exp(
            -0.5 * np.einsum("ikl,ij,jkl->kl", dr, np.linalg.inv(sigma0), dr)
        )

        super().__init__(matrix)

class HarmonicPotential(Potential):
    def __init__(self, size_x : int, size_y : int,
                  k : float, r0 : tuple[int, int]):
        """Quadratic potential with minimum at r0 and strength constant k."""
        _x = np.arange(size_x)
        _y = np.arange(size_y)

        x, y = np.meshgrid(_x, _y, indexing="ij")

        dx = x - r0[0]
        dy = y - r0[1]

        matrix = 0.5 * k * (dx ** 2 + dy ** 2)

        super().__init__(matrix)