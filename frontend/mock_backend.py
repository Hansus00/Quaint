# ==============================================================================
# ### --- FILE mock_backend.py --- ###
# ==============================================================================

from typing import List
import numpy as np


class QuantumMockBackend:
    """
    Simulation mock backend that maintains physical states and generates basic analytical wave frames.
    Used during early development as a lightweight placeholder for the numerical solver.
    """

    def __init__(self, x_coarse: np.ndarray, y_coarse: np.ndarray, total_frames: int) -> None:
        self.x_coarse: np.ndarray = x_coarse
        self.y_coarse: np.ndarray = y_coarse
        self.total_frames: int = total_frames

        self.size_x: int = len(x_coarse)
        self.size_y: int = len(y_coarse)

        # Initialize default physical states
        self.potential_coarse: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=float)
        self.r0_physical: np.ndarray = np.array([self.x_coarse[0], self.y_coarse[0]], dtype=float)
        self.k0: np.ndarray = np.array([0.0, 0.0])
        self.sigma_matrix: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.mass: float = 1.0

    def update_setup(
        self,
        potential_coarse: np.ndarray,
        r0_indices: np.ndarray,
        k0: np.ndarray,
        sigma_matrix: np.ndarray,
        mass: float,
    ) -> None:
        """
        Updates internal physical parameters based on GUI canvas selection.
        """
        self.potential_coarse = potential_coarse

        idx_x: int = int(np.clip(r0_indices[0], 0, self.size_x - 1))
        idx_y: int = int(np.clip(r0_indices[1], 0, self.size_y - 1))

        self.r0_physical = np.array([self.x_coarse[idx_x], self.y_coarse[idx_y]])
        self.k0 = np.array([k0[0], k0[1]])

        self.sigma_matrix = sigma_matrix
        self.mass = mass

    def calculate_all_frames(self) -> List[np.ndarray]:
        """
        Calculates and aggregates analytical snapshots for all animation frames.
        """
        wave_frames: List[np.ndarray] = []
        for t in range(self.total_frames):
            psi = self.get_frame(t)
            wave_frames.append(psi)

        return wave_frames

    def get_frame(self, t: int) -> np.ndarray:
        """
        Generates an ideal moving 2D Gaussian wave packet snapshot analytically.
        """
        X, Y = np.meshgrid(self.x_coarse, self.y_coarse, indexing="ij")

        time_step: float = 0.1
        current_rx: float = self.r0_physical[0] + (self.k0[0] / self.mass) * t * time_step
        current_ry: float = self.r0_physical[1] + (self.k0[1] / self.mass) * t * time_step

        sx: float = self.sigma_matrix[0, 0] if self.sigma_matrix[0, 0] > 0 else 1.0
        sy: float = self.sigma_matrix[1, 1] if self.sigma_matrix[1, 1] > 0 else 1.0

        envelope: np.ndarray = np.exp(
            -((X - current_rx) ** 2) / (2 * sx ** 2) - ((Y - current_ry) ** 2) / (2 * sy ** 2)
        )

        phase: np.ndarray = np.exp(1j * (self.k0[0] * X + self.k0[1] * Y - t * 0.5))

        return envelope * phase