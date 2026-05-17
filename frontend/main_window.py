# ==============================================================================
# ### --- FILE frontend/main_window.py --- ###
# ==============================================================================

from typing import Any

import numpy as np
from backend.Potential import InfiniteWellPotential, Potential
from backend.Solver import SSFM, Constant, CrankNicolson
from backend.StationaryWaveFunc import GaussianPacket
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from .animation_controls_widget import AnimationControlsWidget
from .animation_widget import AnimationWidget
from .settings import Settings
from .setup_drawer import SetupDrawer


class MainWindow(QMainWindow):
    """
    Main Application Window linking the physical data, 3D visualization, and UI controls.
    Coordinates the simulation state and passes data to the rendering widgets.
    """

    # --- Class Fields ---
    size_coarse_x: int
    size_coarse_y: int
    z_potential_offset: float
    z_scale: float
    fine_grid_scale: int
    z_potential_scale: float
    brightness_multiplier: float
    total_frames: int
    fps: int
    aspect_ratio: float
    x_limit: float
    y_limit: float
    x_coarse: np.ndarray
    y_coarse: np.ndarray
    wave_frames: list
    initial_potential: Potential
    initial_wavefunc: GaussianPacket
    current_potential_array: np.ndarray
    current_r0: np.ndarray
    current_k0: np.ndarray
    current_sigma: np.ndarray
    current_mass: float
    current_method: str
    animation_widget: AnimationWidget
    controls: AnimationControlsWidget
    simulation: Any

    def __init__(
        self, size_x: int = 60, size_y: int = 50, z_potential_offset: float = 0.1
    ) -> None:
        """
        Initializes the main window and default simulation states.

        Args:
            size_x (int): Horizontal resolution of the internal physical grid.
            size_y (int): Vertical resolution of the internal physical grid.
            z_potential_offset (float): Visual downward shift multiplier for the potential mesh.
        """
        super().__init__()
        self.setWindowTitle("3D Wave Function & Potential Simulation")
        self.resize(950, 750)

        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.z_potential_offset = z_potential_offset
        self.z_scale = 15.0
        self.fine_grid_scale = 4
        self.z_potential_scale = 0.07
        self.brightness_multiplier = 25.0

        self.total_frames = 150
        self.fps = 30

        self.aspect_ratio = self.size_coarse_y / self.size_coarse_x
        self.x_limit = 10.0
        self.y_limit = 10.0 * self.aspect_ratio

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self.wave_frames = []
        self.initial_potential = InfiniteWellPotential(
            self.size_coarse_x, self.size_coarse_y
        )
        self.initial_wavefunc = GaussianPacket(
            r0=np.array([self.size_coarse_x / 2, self.size_coarse_y / 2]),
            k0=np.array([0.0, 0.0]),
            sigma0=np.array([[1.0, 0.0], [0.0, 1.0]]),
            mass=1.0,
            size_x=self.size_coarse_x,
            size_y=self.size_coarse_y,
        )

        # Cache for UI state to restore it in the SetupDrawer window upon reopening
        # Potential array flipped for standard 2D UI visual orientation
        self.current_potential_array = self.initial_potential.matrix[:, ::-1].copy()
        self.current_r0 = np.array([self.size_coarse_x / 2, self.size_coarse_y / 2])
        self.current_k0 = np.array([0.0, 0.0])
        self.current_sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.current_mass = 1.0

        # Default simulation method
        self.current_method = "Crank-Nicolson"
        self.switch_simulation_method(self.current_method)

        self._setup_ui()
        # Populate the 3D potential mesh immediately on startup to prevent NoneType crash
        self.animation_widget.update_potential(self.initial_potential.matrix)
        self.calculate_all_frames()
        self.update_simulation(0)

    def _setup_ui(self) -> None:
        """
        Sets up the central widget, 3D viewport, and control panels.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.animation_widget = AnimationWidget(
            self.size_coarse_x,
            self.size_coarse_y,
            self.x_limit,
            self.y_limit,
            self.z_potential_offset,
            self.z_scale,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
        )
        layout.addWidget(self.animation_widget, stretch=1)

        self.controls = AnimationControlsWidget(self.total_frames, self.fps)
        self.controls.frame_changed.connect(self.update_simulation)
        self.controls.open_setup_requested.connect(self.open_setup_drawer)
        self.controls.open_settings_requested.connect(self.open_settings_window)
        self.controls.toggle_potential_requested.connect(
            self.animation_widget.set_potential_visible
        )
        layout.addWidget(self.controls, stretch=0)

    def calculate_all_frames(self) -> None:
        """
        Pre-calculates all simulation frames using the currently selected backend solver
        so that subsequent rendering has zero computational lag.
        """
        self.wave_frames = []
        self.animation_widget.clear_cache()
        self.switch_simulation_method(self.current_method)
        self.wave_frames.append(self.simulation.get_wave_function())

        for _ in range(1, self.total_frames):
            self.simulation.step()
            self.wave_frames.append(self.simulation.get_wave_function())
            # Don't freeze the application UI while the simulation runs its heavy math loop
            QApplication.processEvents()

    def open_settings_window(self) -> None:
        """
        Opens the purely visual playback settings dialog (e.g. brightness, scaling).
        """
        self.controls.pause()
        settings_dialog = Settings(
            self.z_scale,
            self.z_potential_offset,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
            self,
        )
        settings_dialog.settings_saved.connect(self.apply_settings)
        settings_dialog.exec()

    def apply_settings(
        self,
        z_scale: float,
        z_offset: float,
        fine_grid_scale: int,
        z_pot_scale: float,
        brightness: float,
    ) -> None:
        """
        Applies visual settings instantly without recalculating the physics backend.

        Args:
            z_scale (float): Upward multiplier for wave amplitude.
            z_offset (float): Downward shift parameter for potential field.
            fine_grid_scale (int): Number of sub-pixels per physical grid point.
            z_pot_scale (float): Upward multiplier for the drawn potential structure.
            brightness (float): Scalar applied prior to value clip to expose wave tails.
        """
        self.z_scale = z_scale
        self.z_potential_offset = z_offset
        self.fine_grid_scale = fine_grid_scale
        self.z_potential_scale = z_pot_scale
        self.brightness_multiplier = brightness

        self.animation_widget.update_config(
            self.size_coarse_x,
            self.size_coarse_y,
            self.y_limit,
            self.z_potential_offset,
            self.z_scale,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
        )

        self.animation_widget.update_potential(self.initial_potential.matrix)

        # Redraw the current frame with the updated display properties
        self.update_simulation(self.controls.slider.value())

    def open_setup_drawer(self) -> None:
        """
        Opens the canvas drawer for setting up potentials, physical resolutions, and wavepackets.
        """
        self.controls.pause()
        drawer = SetupDrawer(
            current_fps=self.fps,
            current_frames=self.total_frames,
            grid_size_x=self.size_coarse_x,
            grid_size_y=self.size_coarse_y,
            x_limit=self.x_limit,
            y_limit=self.y_limit,
            initial_potential=self.current_potential_array,
            initial_r0=self.current_r0,
            initial_k0=self.current_k0,
            initial_sigma=self.current_sigma,
            initial_mass=self.current_mass,
            initial_method=self.current_method,
            parent=self,
        )
        drawer.setup_saved.connect(self.apply_setup)
        drawer.simulation_changed.connect(self.switch_simulation_method)
        drawer.exec()

    def apply_setup(
        self,
        potential_array: np.ndarray,
        r0: np.ndarray,
        k0: np.ndarray,
        sigma_matrix: np.ndarray,
        mass: float,
        fps: int,
        total_frames: int,
        size_x: int,
        size_y: int,
    ) -> None:
        """
        Applies physics configuration, rebuilds the internal arrays if resolution changes,
        and triggers a complete simulation recalculation.

        Args:
            potential_array (np.ndarray): The 2D numerical mapping of potential energy barriers.
            r0 (np.ndarray): Physical X/Y coordinate starting vector.
            k0 (np.ndarray): Physical X/Y momentum vector defining wave trajectory.
            sigma_matrix (np.ndarray): Covariance matrix defining quantum spatial spread.
            mass (float): Mass of the particle.
            fps (int): Playback speed.
            total_frames (int): Maximum limit of simulation cycles.
            size_x (int): Horizontal computational points.
            size_y (int): Vertical computational points.
        """
        self.fps = fps
        self.total_frames = total_frames
        self.controls.update_settings(fps, total_frames)

        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.aspect_ratio = self.size_coarse_y / self.size_coarse_x
        self.y_limit = 10.0 * self.aspect_ratio

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        # Update cached UI structures directly from the pre-scaled drawer return array
        self.current_potential_array = potential_array.copy()
        self.current_r0 = r0.copy()
        self.current_k0 = k0.copy()
        self.current_sigma = sigma_matrix.copy()
        self.current_mass = mass
        print("Received setup:")
        print(f"k0: {k0}")
        print(f"r0: {r0}")
        print(f"sigma: {sigma_matrix}")
        print(f"mass: {mass}")

        # Flip the potential array along the Y-axis for standard visualization orientation
        potential_coarse = potential_array[:, ::-1]

        self.initial_potential = Potential(potential_coarse)
        self.initial_wavefunc = GaussianPacket(
            self.current_r0,
            self.current_k0,
            self.current_sigma,
            self.current_mass,
            self.size_coarse_x,
            self.size_coarse_y,
        )

        self.animation_widget.update_config(
            self.size_coarse_x,
            self.size_coarse_y,
            self.y_limit,
            self.z_potential_offset,
            self.z_scale,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
        )

        self.calculate_all_frames()
        self.animation_widget.update_potential(self.initial_potential.matrix)
        self.update_simulation(self.controls.slider.value())

    def update_simulation(self, frame_idx: int) -> None:
        """
        Pushes a specific frame from the calculated buffer to the rendering widget.

        Args:
            frame_idx (int): The index in `self.wave_frames` to map to the 3D widget.
        """
        if not self.wave_frames or frame_idx >= len(self.wave_frames):
            return

        psi_coarse = self.wave_frames[frame_idx]
        self.animation_widget.update_wave(psi_coarse)

    def switch_simulation_method(self, method_name: str) -> None:
        """
        Switches the backend solver instance used for calculating the wave evolution.

        Args:
            method_name (str): Dropdown key corresponding to physics solvers (e.g. "Crank-Nicolson").
        """
        self.current_method = method_name
        delta_t: float = 1 / self.fps

        if method_name == "Constant":
            self.simulation = Constant(
                self.initial_potential, self.initial_wavefunc, delta_t
            )
        elif method_name == "Crank-Nicolson":
            self.simulation = CrankNicolson(
                self.initial_potential, self.initial_wavefunc, delta_t
            )
        elif method_name == "SSFM":
            self.simulation = SSFM(
                self.initial_potential, self.initial_wavefunc, delta_t
            )
        else:
            raise ValueError(f"Unknown simulation method: {method_name}")
