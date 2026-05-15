# ==============================================================================
# ### --- FILE main_window.py --- ###
# ==============================================================================
import numpy as np
from animation_controls_widget import AnimationControlsWidget
from animation_widget import AnimationWidget
from mock_backend import QuantumMockBackend
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from settings import Settings
from setup_drawer import SetupDrawer


class MainWindow(QMainWindow):
    """
    Main Application Window linking the data, 3D visualization, and UI controls.
    """

    def __init__(self, size_x=10, size_y=5, z_potential_offset=5):
        super().__init__()
        self.setWindowTitle("3D Wave Function & Potential Simulation")
        self.resize(950, 750)

        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.z_potential_offset = z_potential_offset

        self.total_frames = 150
        self.fps = 30

        self.aspect_ratio = self.size_coarse_y / self.size_coarse_x
        self.x_limit = 10.0
        self.y_limit = 10.0 * self.aspect_ratio

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self.wave_frames = []

        # Backend now manages all physical states internally
        self.backend = QuantumMockBackend(
            self.x_coarse, self.y_coarse, self.total_frames
        )

        self._setup_ui()
        self.calculate_all_frames()
        self.update_simulation(0)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.animation_widget = AnimationWidget(
            self.size_coarse_x,
            self.size_coarse_y,
            self.x_limit,
            self.y_limit,
            self.z_potential_offset,
        )
        layout.addWidget(self.animation_widget, stretch=1)

        self.controls = AnimationControlsWidget(self.total_frames, self.fps)
        self.controls.frame_changed.connect(self.update_simulation)
        self.controls.open_setup_requested.connect(self.open_setup_drawer)
        self.controls.open_settings_requested.connect(self.open_settings_window)
        layout.addWidget(self.controls, stretch=0)

    def calculate_all_frames(self):
        # Delegate the entire frame calculation loop to the backend
        self.wave_frames = self.backend.calculate_all_frames()

    def open_settings_window(self):
        self.controls.pause()
        settings_dialog = Settings(self.fps, self.total_frames, self)
        settings_dialog.settings_saved.connect(self.apply_settings)
        settings_dialog.exec()

    def apply_settings(self, fps, total_frames):
        self.fps = fps
        self.total_frames = total_frames
        self.controls.update_settings(fps, total_frames)

        # Update backend property instead of recreating it so physical states aren't lost
        self.backend.total_frames = self.total_frames
        self.calculate_all_frames()
        self.update_simulation(self.controls.slider.value())

    def open_setup_drawer(self):
        self.controls.pause()
        drawer = SetupDrawer(
            grid_size_x=self.size_coarse_x,
            grid_size_y=self.size_coarse_y,
            x_limit=self.x_limit,
            y_limit=self.y_limit,
            parent=self,
        )
        drawer.setup_saved.connect(self.apply_setup)
        drawer.exec()

    def apply_setup(self, potential_array, r0, k0, sigma_matrix, mass):
        # Flip the potential array along the Y-axis for standard visualization orientation
        potential_coarse = potential_array[:, ::-1]
        print(potential_coarse)

        # Pass pure setup data directly to the backend
        self.backend.update_setup(
            potential_coarse=potential_coarse,
            r0_indices=r0,
            k0=k0,
            sigma_matrix=sigma_matrix,
            mass=mass,
        )

        self.calculate_all_frames()

        # Retrieve the updated potential array to draw
        self.animation_widget.update_potential(self.backend.potential_coarse)
        self.update_simulation(self.controls.slider.value())

    def update_simulation(self, frame_idx):
        if not self.wave_frames or frame_idx >= len(self.wave_frames):
            return

        psi_coarse = self.wave_frames[frame_idx]
        self.animation_widget.update_wave(psi_coarse)
