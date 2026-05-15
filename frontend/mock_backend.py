# ==============================================================================
# ### --- FILE mock_backend.py --- ###
# ==============================================================================
import numpy as np

class QuantumMockBackend:
    """
    Simulation backend that maintains physical states and processes wave frames.
    """
    def __init__(self, x_coarse, y_coarse, total_frames):
        self.x_coarse = x_coarse
        self.y_coarse = y_coarse
        self.total_frames = total_frames
        
        self.size_x = len(x_coarse)
        self.size_y = len(y_coarse)

        # Initialize default physical states
        self.potential_coarse = np.zeros((self.size_x, self.size_y), dtype=float)
        self.r0_physical = np.array([self.x_coarse[0], self.y_coarse[0]], dtype=float)
        self.k0 = np.array([0.0, 0.0])
        self.sigma_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.mass = 1.0

    def update_setup(self, potential_coarse, r0_indices, k0, sigma_matrix, mass):
        """
        Updates internal physical parameters. 
        Directly maps natural grid indices to physical coordinates without inversions.
        """
        self.potential_coarse = potential_coarse
        
        # REMOVED Y-axis flipping logic. Direct Cartesian mapping.
        idx_x = int(np.clip(r0_indices[0], 0, self.size_x - 1))
        idx_y = int(np.clip(r0_indices[1], 0, self.size_y - 1))
        
        self.r0_physical = np.array([self.x_coarse[idx_x], self.y_coarse[idx_y]])
        
        # Keep k0 exactly as provided by the UI
        self.k0 = np.array([k0[0], k0[1]]) 
        
        self.sigma_matrix = sigma_matrix
        self.mass = mass

    def calculate_all_frames(self):
        wave_frames = []
        for t in range(self.total_frames):
            psi = self.get_frame(t)
            wave_frames.append(psi)
            
        return wave_frames

    def get_frame(self, t):
        """
        Generates a moving 2D Gaussian wave packet based on internal physical states.
        """
        X, Y = np.meshgrid(self.x_coarse, self.y_coarse, indexing="ij")

        # Calculate position over time
        time_step = 0.1  
        current_rx = self.r0_physical[0] + (self.k0[0] / self.mass) * t * time_step
        current_ry = self.r0_physical[1] + (self.k0[1] / self.mass) * t * time_step

        # Extract standard deviations 
        sx = self.sigma_matrix[0, 0] if self.sigma_matrix[0, 0] > 0 else 1.0
        sy = self.sigma_matrix[1, 1] if self.sigma_matrix[1, 1] > 0 else 1.0

        # Construct Gaussian envelope
        envelope = np.exp(
            -((X - current_rx) ** 2) / (2 * sx ** 2)
            - ((Y - current_ry) ** 2) / (2 * sy ** 2)
        )

        # Construct Phase
        phase = np.exp(1j * (self.k0[0] * X + self.k0[1] * Y - t * 0.5))

        return envelope * phase
