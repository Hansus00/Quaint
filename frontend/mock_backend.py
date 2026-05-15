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
        
        Note: potential and psi_prev are accepted to maintain interface 
        compatibility but are completely ignored to simulate a free-space packet.
        """
        # Time step scaling factor for smooth propagation across frame updates
        dt = 0.08
        current_time = t * dt

        # 1. Determine velocity from momentum (k0) and mass: v = k / m
        vx = k0[0] / mass
        vy = k0[1] / mass

        # 2. Calculate current center position of the packet
        x_pos = r0[0] + vx * current_time
        y_pos = r0[1] + vy * current_time

        # 3. Safely parse spatial width variances out of sigma_matrix
        try:
            if np.ndim(sigma_matrix) == 2:
                sig_x = sigma_matrix[0, 0]
                sig_y = sigma_matrix[1, 1]
            else:
                sig_x = sigma_matrix[0]
                sig_y = sigma_matrix[1]
        except (TypeError, IndexError):
            sig_x = sig_y = float(sigma_matrix) if isinstance(sigma_matrix, (int, float)) else 1.0

        # Avoid zero division errors
        sig_x = max(sig_x, 1e-5)
        sig_y = max(sig_y, 1e-5)

        # 4. Construct the Gaussian spatial envelope
        envelope = np.exp(-0.5 * (((self.X - x_pos) / sig_x) ** 2 + ((self.Y - y_pos) / sig_y) ** 2))

        # 5. Apply the quantum phase factor: exp(1j * (k·r - E·t))
        # Kinetic energy: E = |k|^2 / (2 * mass)
        k_squared = k0[0]**2 + k0[1]**2
        energy = k_squared / (2.0 * mass)
        phase = np.exp(1j * (k0[0] * self.X + k0[1] * self.Y - energy * current_time))

        # Combine into the finalized free wave packet
        psi_new = envelope * phase

        # 6. Normalize the wave function amplitude
        max_val = np.max(np.abs(psi_new))
        if max_val > 0:
            psi_new /= max_val

        return psi_new
