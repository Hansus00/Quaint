# ==============================================================================
# ### --- FILE frontend/main_window.py --- ###
# ==============================================================================

from typing import Any, Optional, cast

import numpy as np
from backend.Potential import InfiniteWellPotential, Potential
from backend.Solver import SSFM, Constant, CrankNicolson
from backend.StationaryWaveFunc import GaussianPacket
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QMessageBox

from .animation_controls_widget import AnimationControlsWidget
from .animation_widget import AnimationWidget
from .settings import Settings
from .setup_drawer import SetupDrawer
from .simulation_thread import SimulationThread
from .warning_handler import WarningCaptureHandler
import logging

logger = logging.getLogger(__name__)


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
    potential_alpha: float
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

    current_delta_t: float
    current_steps_per_frame: int
    current_wall_height: float

    animation_widget: AnimationWidget
    controls: AnimationControlsWidget
    simulation: Any
    _settings_dialog: Optional[Settings]
    _setup_drawer: Optional[SetupDrawer]
    # TODO: move to a dataclass
    _pending_setup: Optional[dict[str, Any]]
    _calculation_cancelled: bool
    worker: Optional[SimulationThread]

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

        self.x_limit = 10.0
        self.y_limit = 10.0
        self.current_grid_step = 0.2

        self.size_coarse_x = int(self.x_limit / self.current_grid_step)
        self.size_coarse_y = int(self.y_limit / self.current_grid_step)
        self.aspect_ratio = self.y_limit / self.x_limit
        self.z_potential_offset = z_potential_offset
        self.z_scale = 150.0
        self.fine_grid_scale = 4
        self.z_potential_scale = 0.07
        self.brightness_multiplier = 25.0
        self.potential_alpha = 0.4  # Default opacity level

        self.total_frames = 150
        self.fps = 30

        self.current_delta_t = 0.001
        self.current_steps_per_frame = 50
        self.current_wall_height = 50.0

        self.aspect_ratio = self.size_coarse_y / self.size_coarse_x
        self.x_limit = 10.0
        self.y_limit = 10.0 * self.aspect_ratio

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self.wave_frames = []
        self.initial_potential = InfiniteWellPotential(
            self.size_coarse_x, self.size_coarse_y
        )
        self.initial_potential = self._coarse_potential_from_drawer(
            self.initial_potential.matrix, self.current_wall_height
        )
        self.initial_wavefunc = GaussianPacket(
            r0=(self.size_coarse_x // 2, self.size_coarse_y // 2),
            k0=np.array([0.0, 0.0]),
            sigma0=np.array([[4.0, 0.0], [0.0, 4.0]]),
            mass=1.0,
            size_x=self.size_coarse_x,
            size_y=self.size_coarse_y,
        )

        # Cache for UI state to restore it in the SetupDrawer window upon reopening
        # Potential array flipped for standard 2D UI visual orientation
        self.current_potential_array = self.initial_potential.matrix[:, ::-1].copy()
        self.current_r0 = np.array([self.size_coarse_x / 2, self.size_coarse_y / 2])
        self.current_k0 = np.array([0.0, 0.0])
        self.current_sigma = np.array([[4.0, 0.0], [0.0, 4.0]])
        self.current_mass = 1.0

        # Default simulation method
        self.current_method = "Crank-Nicolson"

        self._settings_dialog = None
        self._setup_drawer = None
        self._pending_setup = None
        self._calculation_cancelled = False
        self.worker = None

        self._setup_ui()
        # Populate the 3D potential mesh immediately on startup to prevent NoneType crash
        self.animation_widget.update_potential(self.initial_potential.matrix)
        self.calculate_all_frames()

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
            self.potential_alpha,
        )
        layout.addWidget(self.animation_widget, stretch=1)

        self.controls = AnimationControlsWidget(self.total_frames, self.fps)
        self.controls.frame_changed.connect(self.update_simulation)
        self.controls.open_setup_requested.connect(self.open_setup_drawer)
        self.controls.open_settings_requested.connect(self.open_settings_window)
        self.controls.reset_camera_requested.connect(self.animation_widget.reset_camera)
        self.controls.toggle_potential_requested.connect(
            self.animation_widget.set_potential_visible
        )
        self.controls.stop_calculation_requested.connect(self.stop_calculation)
        layout.addWidget(self.controls, stretch=0)

    def _coarse_potential_from_drawer(
        self, potential_array: np.ndarray, wall_height: float
    ) -> Potential:
        """Convert drawer potential (UI orientation) to backend coarse matrix."""
        potential_coarse = potential_array[:, ::-1].copy()
        potential_coarse[0, :] = wall_height
        potential_coarse[-1, :] = wall_height
        potential_coarse[:, 0] = wall_height
        potential_coarse[:, -1] = wall_height
        return Potential(potential_coarse)

    def _instantiate_solver(
        self,
        method_name: str,
        potential: Potential,
        wavefunc: GaussianPacket,
        delta_t: float,
        grid_step: float,
    ) -> Any:

        capture_handler = WarningCaptureHandler()
        solver_logger = logging.getLogger("backend.Solver")
        solver_logger.addHandler(capture_handler)

        if method_name == "Constant":
            simulation = Constant(potential, wavefunc, delta_t, grid_step=grid_step)
        elif method_name == "Crank-Nicolson":
            simulation = CrankNicolson(
                potential, wavefunc, delta_t, grid_step=grid_step
            )
        elif method_name == "SSFM":
            simulation = SSFM(potential, wavefunc, delta_t, grid_step=grid_step)
        else:
            raise ValueError(f"Unknown simulation method: {method_name}")

        # TODO: fix typing (_Solver has no field stability_warnings)
        simulation.stability_warnings = capture_handler.captured_warnings

        solver_logger.removeHandler(capture_handler)
        return simulation

    def _simulation_from_pending(self, pending: dict[str, Any]) -> Any:
        """Build a solver for a pending setup without mutating committed state."""
        potential = self._coarse_potential_from_drawer(
            pending["potential_array"], pending["wall_height"]
        )
        wavefunc = GaussianPacket(
            cast(tuple[int, int], tuple(np.asarray(pending["r0"]).astype(int))),
            pending["k0"],
            pending["sigma_matrix"],
            pending["mass"],
            pending["size_x"],
            pending["size_y"],
        )
        return self._instantiate_solver(
            pending["method"],
            potential,
            wavefunc,
            pending["delta_t"],
            grid_step=pending["grid_step"],
        )

    def _commit_pending_setup(self) -> None:
        """Apply a successful calculation's pending setup to the live simulation."""
        pending = self._pending_setup
        if pending is None:
            return

        self.fps = pending["fps"]
        self.total_frames = pending["total_frames"]
        self.controls.update_settings(self.fps, self.total_frames)

        self.current_delta_t = pending["delta_t"]
        self.current_steps_per_frame = pending["steps_per_frame"]
        self.current_wall_height = pending["wall_height"]

        self.size_coarse_x = pending["size_x"]
        self.size_coarse_y = pending["size_y"]

        self.x_limit = pending["x_limit"]
        self.y_limit = pending["y_limit"]
        self.current_grid_step = pending["grid_step"]
        self.aspect_ratio = self.y_limit / self.x_limit

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self.current_potential_array = pending["potential_array"].copy()
        self.current_r0 = np.asarray(pending["r0"], dtype=np.float64).copy()
        self.current_k0 = np.asarray(pending["k0"], dtype=np.float64).copy()
        self.current_sigma = np.asarray(
            pending["sigma_matrix"], dtype=np.float64
        ).copy()
        self.current_mass = pending["mass"]
        self.current_method = pending["method"]

        self.initial_potential = self._coarse_potential_from_drawer(
            self.current_potential_array, self.current_wall_height
        )
        self.initial_wavefunc = GaussianPacket(
            cast(tuple[int, int], tuple(self.current_r0.astype(int))),
            self.current_k0,
            self.current_sigma,
            self.current_mass,
            self.size_coarse_x,
            self.size_coarse_y,
        )

        # Solver was already created for the worker in calculate_all_frames;
        # re-instantiating here would log "New simulation" twice per save.
        if self.worker is not None:
            self.simulation = self.worker.simulation

        self.animation_widget.update_config(
            self.size_coarse_x,
            self.size_coarse_y,
            self.x_limit,
            self.y_limit,
            self.z_potential_offset,
            self.z_scale,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
            self.potential_alpha,
        )
        self._pending_setup = None

    def calculate_all_frames(self) -> None:
        """
        Initiates a new thread to calculate the entire simulation sequence based on the current setup and method.
        This is necessary to prevent UI freezing during heavy computations.
        """
        self._calculation_cancelled = False

        if self._pending_setup is not None:
            sim = self._simulation_from_pending(self._pending_setup)
            total_frames = self._pending_setup["total_frames"]
            steps_per_frame = self._pending_setup["steps_per_frame"]
        else:
            self.animation_widget.clear_cache()
            self.switch_simulation_method(self.current_method)
            sim = self.simulation
            total_frames = self.total_frames
            steps_per_frame = self.current_steps_per_frame

        if hasattr(sim, "stability_warnings") and sim.stability_warnings:
            warning_text = "\n\n".join(sim.stability_warnings)
            QMessageBox.warning(
                self,
                "Simulation Stability Warning",
                f"The physical parameters might cause the simulation to become unstable "
                f"or mathematically inaccurate:\n\n{warning_text}",
            )
            return

        self.controls.enter_calculating_mode()
        self.controls.time_label.setText("Calculating...")

        self.worker = SimulationThread(sim, total_frames, steps_per_frame)
        self.worker.calculation_finished.connect(self.on_calculation_finished)
        self.worker.calculation_cancelled.connect(self.on_calculation_cancelled)
        self.worker.start()

    def stop_calculation(self) -> None:
        """Stop an in-flight run; committed state is unchanged because setup applies on finish."""
        if self.worker is None or not self.worker.isRunning():
            return

        self._calculation_cancelled = True
        self._pending_setup = None
        self.worker.request_cancel()
        self.controls.exit_calculating_mode()
        self.controls.time_label.setText(f"Time: {self.controls.slider.value()}")

    def on_calculation_cancelled(self) -> None:
        self._calculation_cancelled = False

    def on_calculation_finished(self, generated_frames: list) -> None:
        """
        Receives the calculated frames from the worker thread.
        Dynamically sizes the animation cache based on remaining system RAM.
        """
        if self._calculation_cancelled:
            self._calculation_cancelled = False
            return

        if self._pending_setup is not None:
            self.animation_widget.clear_cache()
            self._commit_pending_setup()

        self.wave_frames = generated_frames

        # --- DYNAMIC CACHE SIZING AFTER PHYSICS ALLOCATION ---
        try:
            import psutil

            mem_available = psutil.virtual_memory().available
        except ImportError:
            # Fallback if psutil is not available, assume 16GB free memory
            mem_available = 16 * 1024 * 1024 * 1024

        # Dedicate up to 50% of the remaining free RAM to the OpenGL rendering cache
        cache_memory_allowance = mem_available * 0.50

        # Calculate memory footprint of a single OpenGL cached frame:
        # verts (float32, 3 cols = 12 bytes) + rgba (float32, 4 cols = 16 bytes) = 28 bytes per fine grid point
        nx_fine = self.animation_widget.size_fine_x
        ny_fine = self.animation_widget.size_fine_y
        bytes_per_cached_frame = (nx_fine * ny_fine) * 28

        if bytes_per_cached_frame > 0:
            safe_cache_limit = int(cache_memory_allowance / bytes_per_cached_frame)
            # Cap the cache at `self.total_frames` (no need to cache more than exists)
            # Ensure a minimum of 10 frames to avoid breaking playback
            final_cache_limit = max(10, min(safe_cache_limit, self.total_frames))
            self.animation_widget.max_cache_size = final_cache_limit
        else:
            self.animation_widget.max_cache_size = self.total_frames

        logger.info(
            f"Physics complete. Set UI cache limit to: {self.animation_widget.max_cache_size} frames."
        )

        self.controls.exit_calculating_mode()
        self.controls.time_label.setText(f"Time: {self.controls.slider.value()}")

        # Updating the simulation
        self.animation_widget.update_potential(self.initial_potential.matrix)
        self.update_simulation(self.controls.slider.value())

    def _raise_auxiliary_window(self, window: QWidget) -> None:
        """Brings a non-modal auxiliary window to the front without blocking the main UI."""
        window.show()
        main_hw = self.windowHandle()
        child_hw = window.windowHandle()
        if main_hw is not None and child_hw is not None:
            child_hw.setTransientParent(main_hw)
        window.raise_()
        window.activateWindow()

    def open_settings_window(self) -> None:
        """
        Opens the purely visual playback settings dialog (e.g. brightness, scaling).
        Non-modal: playback continues and the main window stays fully interactive.
        """
        if self._settings_dialog is not None:
            self._raise_auxiliary_window(self._settings_dialog)
            return

        settings_dialog = Settings(
            self.z_scale,
            self.z_potential_offset,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
            self.potential_alpha,
            self,
        )
        settings_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        settings_dialog.settings_saved.connect(self.apply_settings)
        settings_dialog.destroyed.connect(self._on_settings_dialog_destroyed)
        self._settings_dialog = settings_dialog
        self._raise_auxiliary_window(settings_dialog)

    def _on_settings_dialog_destroyed(self, _obj: Optional[QWidget] = None) -> None:
        self._settings_dialog = None

    def apply_settings(
        self,
        z_scale: float,
        z_offset: float,
        fine_grid_scale: int,
        z_pot_scale: float,
        brightness: float,
        potential_alpha: float,
    ) -> None:
        """
        Applies visual settings instantly without recalculating the physics backend.

        Args:
            z_scale (float): Upward multiplier for wave amplitude.
            z_offset (float): Downward shift parameter for potential field.
            fine_grid_scale (int): Number of sub-pixels per physical grid point.
            z_pot_scale (float): Upward multiplier for the drawn potential structure.
            brightness (float): Scalar applied prior to value clip to expose wave tails.
            potential_alpha (float): Transparency multiplier for the potential 3D mesh.
        """
        self.z_scale = z_scale
        self.z_potential_offset = z_offset
        self.fine_grid_scale = fine_grid_scale
        self.z_potential_scale = z_pot_scale
        self.brightness_multiplier = brightness
        self.potential_alpha = potential_alpha

        self.animation_widget.update_config(
            self.size_coarse_x,
            self.size_coarse_y,
            self.x_limit,
            self.y_limit,
            self.z_potential_offset,
            self.z_scale,
            self.fine_grid_scale,
            self.z_potential_scale,
            self.brightness_multiplier,
            self.potential_alpha,
        )

        self.animation_widget.update_potential(self.initial_potential.matrix)

        # Redraw the current frame with the updated display properties
        self.update_simulation(self.controls.slider.value())

    def open_setup_drawer(self) -> None:
        """
        Opens the canvas drawer for setting up potentials, physical resolutions, and wavepackets.
        Non-modal: playback continues and the main window stays fully interactive.
        """
        if self._setup_drawer is not None:
            self._raise_auxiliary_window(self._setup_drawer)
            return

        drawer = SetupDrawer(
            current_fps=self.fps,
            current_frames=self.total_frames,
            grid_size_x=self.size_coarse_x,
            grid_size_y=self.size_coarse_y,
            x_limit=self.x_limit,
            y_limit=self.y_limit,
            initial_grid_step=self.current_grid_step,
            initial_potential=self.current_potential_array,
            initial_r0=self.current_r0,
            initial_k0=self.current_k0,
            initial_sigma=self.current_sigma,
            initial_mass=self.current_mass,
            initial_method=self.current_method,
            initial_delta_t=self.current_delta_t,
            initial_steps_per_frame=self.current_steps_per_frame,
            initial_wall_height=self.current_wall_height,
            parent=self,
        )
        drawer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        drawer.setup_saved.connect(self.apply_setup)
        drawer.simulation_changed.connect(self.switch_simulation_method)
        drawer.destroyed.connect(self._on_setup_drawer_destroyed)
        self._setup_drawer = drawer
        self._raise_auxiliary_window(drawer)

    def _on_setup_drawer_destroyed(self, _obj: Optional[QWidget] = None) -> None:
        self._setup_drawer = None

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
        delta_t: float,
        steps_per_frame: int,
        wall_height: float,
        x_limit: float,
        y_limit: float,
        grid_step: float,
    ) -> None:
        """
        Applies physics configuration, rebuilds the internal arrays if resolution changes,
        and triggers a complete simulation recalculation.
        Committed state is updated only after the calculation finishes successfully.
        """
        logger.info("Received setup:")
        logger.info(f"k0: {k0}")
        logger.info(f"r0: {r0}")
        logger.info(f"sigma: {sigma_matrix}")
        logger.info(f"mass: {mass}")

        self._pending_setup = {
            "potential_array": potential_array.copy(),
            "r0": r0.copy(),
            "k0": k0.copy(),
            "sigma_matrix": sigma_matrix.copy(),
            "mass": mass,
            "fps": fps,
            "total_frames": total_frames,
            "size_x": size_x,
            "size_y": size_y,
            "delta_t": delta_t,
            "steps_per_frame": steps_per_frame,
            "wall_height": wall_height,
            "method": self.current_method,
            "x_limit": x_limit,
            "y_limit": y_limit,
            "grid_step": grid_step,
        }

        self.calculate_all_frames()

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
        Switches the backend solver instance used for calculating the wave evolution.
        """
        self.current_method = method_name
        self.simulation = self._instantiate_solver(
            method_name,
            self.initial_potential,
            self.initial_wavefunc,
            self.current_delta_t,
            self.current_grid_step,
        )
