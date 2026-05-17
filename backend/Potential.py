from numpy.typing import NDArray
import numpy as np


class Potential:
    matrix: NDArray[np.float128]

    def __init__(self, matrix: NDArray[np.float128]):
        self.matrix = matrix

    def __add__(self, other: "Potential") -> "Potential":
        return Potential(self.matrix + other.matrix)

    def __str__(self) -> str:
        ret = ""
        for k in self.matrix:
            ret += str(k) + "\n"
        return ret


class InfiniteWellPotential(Potential):
    def __init__(self, size_x: int, size_y: int, wall_value: float = 100):
        matrix = np.zeros((size_x, size_y), dtype=np.float128)

        for x in range(0, size_x):
            matrix[x, 0] = wall_value
            matrix[x, size_y - 1] = wall_value

        for y in range(0, size_y):
            matrix[0, y] = wall_value
            matrix[size_x - 1, y] = wall_value

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
    def __init__(self, size_x: int, size_y: int, k: float, r0: tuple[int, int]):
        """Quadratic potential with minimum at r0 and strength constant k."""
        _x = np.arange(size_x)
        _y = np.arange(size_y)

        x, y = np.meshgrid(_x, _y, indexing="ij")

        dx = x - r0[0]
        dy = y - r0[1]

        matrix = 0.5 * k * (dx**2 + dy**2)

        super().__init__(matrix)


class PotentialInsideGrid(Potential):
    """Put smaller potential (left upper corner has position (pos_x, pos_y))
    inside empty frame of bigger one (size_x, size_y), make sure it fits!"""

    def __init__(
        self, size_x: int, size_y: int, pos_x: int, pos_y: int, potential: Potential
    ):
        # Get dimensions of the inner potential
        inner_x, inner_y = potential.matrix.shape

        # Check if the potential fits inside the grid at the given offset
        assert pos_x >= 0 and pos_y >= 0, "Positions must be non-negative."
        assert (
            size_x >= pos_x + inner_x
        ), f"Potential exceeds X boundary: {pos_x + inner_x} > {size_x}"
        assert (
            size_y >= pos_y + inner_y
        ), f"Potential exceeds Y boundary: {pos_y + inner_y} > {size_y}"

        self.matrix = np.zeros((size_x, size_y), dtype=potential.matrix.dtype)
        self.matrix[pos_x : pos_x + inner_x, pos_y : pos_y + inner_y] = potential.matrix


class SharpWShapedPotential(Potential):
    def __init__(self, size_x: int, size_y: int, thickness=3, wall_value: float = 100):
        self.matrix = np.array([[0 for _ in range(size_x)] for _ in range(size_y)])

        center_x = size_x // 2

        # how many on left and right
        offset_left = thickness // 2
        offset_right = thickness - offset_left

        for y in range(size_y):
            # left lower
            x1 = int(y * (center_x / 2) / (size_y - 1)) if size_y > 1 else 0

            # left upper
            x2 = center_x - x1

            # right lower
            x3 = center_x + x1

            # 4. right upper
            x4 = (size_x - 1) - x1

            for x in (x1, x2, x3, x4):
                # thickness
                for dx in range(-offset_left, offset_right):
                    nx = x + dx
                    # safety
                    if 0 <= nx < size_x:
                        self.matrix[y][nx] = wall_value
