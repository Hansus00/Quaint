from numpy.typing import NDArray
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StationaryWaveFunc:
    matrix: NDArray[np.complex128]

    def __init__(
        self,
        matrix: NDArray[np.complex128],
    ):
        self.matrix = matrix
    def total_probability(self) -> float:
        """Returns total probability.
        Calculated as integral of |Psi|^2.
        Should be 1 for normalized wfs."""
        return np.sum(np.abs(self.matrix) ** 2)


class GaussianPacket(StationaryWaveFunc):
    def __init__(
        self,
        r0: tuple[float, float],
        k0: NDArray[np.float64],
        sigma0: NDArray[np.float64],
        size_x: int,
        size_y: int,
    ):
        _x = np.arange(1, size_x + 1)
        _y = np.arange(1, size_y + 1)

        x, y = np.meshgrid(_x, _y, indexing="ij")

        dx = x - r0[0]
        dy = y - r0[1]

        dr = np.stack([dx, dy])

        eigvals = np.linalg.eigvalsh(sigma0)
        signature = (
            np.sum(eigvals > 0),
            np.sum(eigvals < 0),
            np.sum(np.isclose(eigvals, 0)),
        )
        if np.linalg.det(sigma0) == 0:
            logger.critical(
                "sigma0 matrix is a singular matrix, cannot create Gaussian packet"
            )
            return

        if signature != (2, 0, 0):
            logger.critical(
                "sigma0 matrix is not a positive definit, cannot create Gaussian packet"
            )
            return

        matrix = np.exp(
            1j * np.einsum("i,ijk->jk", k0, dr)
            - 0.5 * np.einsum("ikl,ij,jkl->kl", dr, np.linalg.inv(sigma0), dr)
        )

        matrix /= np.sqrt(np.sum(np.abs(matrix) ** 2))
        super().__init__(matrix)
