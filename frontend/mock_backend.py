# ==============================================================================
# ### --- FILE mock_backend.py --- ###
# ==============================================================================
import numpy as np


class QuantumMockBackend:
    """
    Simulates physical dynamics. Operates strictly on backend arrays.
    """

    def __init__(self, x, y, total_frames):
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.total_frames = total_frames

    def get_frame(self, psi_prev, t, potential, r0, k0, sigma_matrix, mass):
        """
        Accepts the previous wave matrix, time, potential, and initial conditions.
        """
        time_factor = t * 2 * np.pi / self.total_frames

        # Integrate r0 and k0 into the mock calculation to show them affecting the backend
        x_pos = r0[0] + k0[0] * t * 0.1 + 2.0 * np.sin(time_factor)
        y_pos = r0[1] + k0[1] * t * 0.1 + 2.0 * np.cos(time_factor)

        # Apply sigma0 to envelope width
        envelope = np.exp(-0.5 * ((self.X - x_pos) ** 2 + (self.Y - y_pos) ** 2))
        phase = np.exp(1j * (k0[0] * self.X + k0[1] * self.Y - 2.0 * time_factor))

        interaction_effect = np.exp(-potential * 3.0)
        psi_new = envelope * phase * interaction_effect + (psi_prev * 0.05)

        max_val = np.max(np.abs(psi_new))
        if max_val > 0:
            psi_new /= max_val

        return psi_new
