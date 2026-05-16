# ==============================================================================
# ### --- FILE frontend/main_window.py --- ###
# ==============================================================================
import numpy as np
from .animation_controls_widget import AnimationControlsWidget
from .animation_widget import AnimationWidget
from backend.Potential import InfiniteWellPotential, Potential
from backend.Solver import SSFM, Constant, CrankNicolson
from backend.StationaryWaveFunc import GaussianPacket
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from .settings import Settings
from .setup_drawer import SetupDrawer


class MainWindow(QMainWindow):
    """
    Main Application Window linking the physical data, 3D visualization, and UI controls.
    Coordinates the simulation state and passes data to the rendering widgets.
    """

    def __init__(
        self, size_x: int = 60, size_y: int = 70, z_potential_offset: float = 0.1
    ) -> None:
        """
        Initializes the main window and default simulation states.

        Args:
            size_x (int): Number of grid points along the X-axis.
            size_y (int): Number of grid points along the Y-axis.
            z_potential_offset (int): Visual offset for rendering the potential.
        """
        super().__init__()
        self.setWindowTitle("3D Wave Function & Potential Simulation")
        self.resize(950, 750)

        self.size_coarse_x: int = size_x
        self.size_coarse_y: int = size_y
        self.z_potential_offset: float = z_potential_offset

        self.total_frames: int = 150
        self.fps: int = 30

        self.aspect_ratio: float = self.size_coarse_y / self.size_coarse_x
        self.x_limit: float = 10.0
        self.y_limit: float = 10.0 * self.aspect_ratio

        self.x_coarse: np.ndarray = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse: np.ndarray = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self.wave_frames: list = []
        self.initial_potential: Potential = InfiniteWellPotential(
            self.size_coarse_x, self.size_coarse_y
        )
        self.initial_wavefunc: GaussianPacket = GaussianPacket(
            r0=np.array([self.size_coarse_x / 2, self.size_coarse_y / 2]),
            k0=np.array([0.0, 0.0]),
            sigma0=np.array([[1.0, 0.0], [0.0, 1.0]]),
            mass=1.0,
            size_x=self.size_coarse_x,
            size_y=self.size_coarse_y,
        )

        # Cache for UI state to restore it in the SetupDrawer window upon reopening
        self.current_potential_array: np.ndarray = np.zeros((self.size_coarse_x, self.size_coarse_y))
        self.current_r0: np.ndarray = np.array([self.size_coarse_x / 2, self.size_coarse_y / 2])
        self.current_k0: np.ndarray = np.array([0.0, 0.0])
        self.current_sigma: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.current_mass: float = 1.0

        # Default simulation method
        self.current_method: str = "Crank-Nicolson"
        self.switch_simulation_method(self.current_method)

        self._setup_ui()
        self.calculate_all_frames()
        self.update_simulation(0)

    def _setup_ui(self) -> None:
        """Sets up the central widget, 3D viewport, and control panels."""
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

    def calculate_all_frames(self) -> None:
        """
        Pre-calculates all simulation frames using the currently selected backend solver.
        """
        self.wave_frames = []

        self.animation_widget.clear_cache()

        self.switch_simulation_method(self.current_method)
        self.wave_frames.append(self.simulation.get_wave_function())

        for _ in range(1, self.total_frames):
            self.simulation.step()
            self.wave_frames.append(self.simulation.get_wave_function())

    def open_settings_window(self) -> None:
        """Opens the playback settings dialog."""
        self.controls.pause()
        settings_dialog = Settings(self.fps, self.total_frames, self)
        settings_dialog.settings_saved.connect(self.apply_settings)
        settings_dialog.exec()

    def apply_settings(self, fps: int, total_frames: int) -> None:
        """Applies new playback settings and recalculates frames if needed."""
        self.fps = fps
        self.total_frames = total_frames
        self.controls.update_settings(fps, total_frames)

        self.calculate_all_frames()
        self.update_simulation(self.controls.slider.value())

    def open_setup_drawer(self) -> None:
        """Opens the canvas drawer for setting up potentials and wavepackets."""
        self.controls.pause()
        drawer = SetupDrawer(
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
    ) -> None:
        """
        Applies the physical setup generated by the SetupDrawer.

        Args:
            potential_array (np.ndarray): 2D array representing the potential landscape.
            r0 (np.ndarray): Initial position vector.
            k0 (np.ndarray): Initial momentum vector.
            sigma_matrix (np.ndarray): 2x2 covariance matrix for the Gaussian packet.
            mass (float): Particle mass.
        """
        # Cache the latest UI settings to be restored when SetupDrawer is opened again
        self.current_potential_array = potential_array.copy()
        self.current_r0 = r0.copy()
        self.current_k0 = k0.copy()
        self.current_sigma = sigma_matrix.copy()
        self.current_mass = mass

        # Flip the potential array along the Y-axis for standard visualization orientation
        potential_coarse = potential_array[:, ::-1]

        self.initial_potential = Potential(potential_coarse)
        self.initial_wavefunc = GaussianPacket(
            r0, k0, sigma_matrix, mass, self.size_coarse_x, self.size_coarse_y
        )
        self.calculate_all_frames()

        # Retrieve the updated potential array to draw
        self.animation_widget.update_potential(self.initial_potential.matrix)
        self.update_simulation(self.controls.slider.value())

    def update_simulation(self, frame_idx: int) -> None:
        """
        Pushes a specific frame from the calculated buffer to the rendering widget.
        """
        if not self.wave_frames or frame_idx >= len(self.wave_frames):
            return

        psi_coarse = self.wave_frames[frame_idx]
        self.animation_widget.update_wave(psi_coarse)

    def switch_simulation_method(self, method_name: str) -> None:
        """
        Switches the backend solver used for calculating the wave evolution.

        Args:
            method_name (str): The name of the method to use (e.g., "Constant", "Crank-Nicolson").
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
            # The SSFM module from backend throws NotImplementedError, leaving it as is for now
            self.simulation = SSFM()
        else:
            raise ValueError(f"Unknown simulation method: {method_name}")
