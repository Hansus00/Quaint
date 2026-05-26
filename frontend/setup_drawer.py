# ==============================================================================
# ### --- FILE frontend/setup_drawer.py --- ###
# ==============================================================================

from typing import Optional

import numpy as np
from backend.Potential import (
    EmbeddedPotential,
    GaussianBumpPotential,
    HarmonicPotential,
    Potential,
    WShaped,
)
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from backend.Params import Params, SolverType, WellType
from .canvas_widget import CanvasWidget, AspectRatioContainer


# Conversion factor between an arrow's grid-cell displacement and the physical
# wavevector magnitude. With 1.0 a drag of N grid cells encodes |k| = N, which
# is grid-resolution independent and yields exact round-trips through the
# (canvas <-> Params) boundary.
K_GRID_FACTOR: float = 0.01


class SetupDrawer(QDialog):
    """
    A comprehensive physics configuration dialogue interface linking user sketches to simulation parameters.

    This window encapsulates input form layout parameters (discretization grids, playback rates, particle masses)
    and binds them alongside an interactive canvas element. Upon closure via submission, it performs
    coordinate mapping and grayscale matrix translation to dispatch structured raw simulation data fields
    back to the primary 3D visualization window engine.
    """

    # -- Class Fields --
    # Emits: (method_name)
    simulation_changed = pyqtSignal(str)

    # Emits: (potential_matrix, r0, k0, sigma_matrix, mass, fps, total_frames, size_x, size_y, delta_t, steps_per_frame, wall_height)
    setup_saved = pyqtSignal(
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        int,
        int,
        int,
        float,
        int,
        float,
    )

    current_fps: int
    current_frames: int
    grid_size_x: int
    grid_size_y: int
    x_limit: float
    y_limit: float
    initial_sigma: Optional[np.ndarray]
    initial_mass: float
    initial_method: str
    initial_delta_t: float
    initial_steps_per_frame: int
    initial_wall_height: float

    canvas: CanvasWidget
    canvas_container: AspectRatioContainer
    simulation_menu: QComboBox
    preset_menu: QComboBox
    fps_input: QSpinBox
    frames_input: QSpinBox
    size_x_input: QSpinBox
    size_y_input: QSpinBox

    delta_t_input: QDoubleSpinBox
    steps_per_frame_input: QSpinBox
    wall_height_input: QDoubleSpinBox

    update_grid_btn: QPushButton
    radio_brush: QRadioButton
    radio_eraser: QRadioButton
    radio_wave: QRadioButton
    sig_xx_input: QDoubleSpinBox
    sig_xy_input: QDoubleSpinBox
    sig_yy_input: QDoubleSpinBox
    mass_input: QDoubleSpinBox
    brush_strength_label: QLabel
    brush_strength_slider: QSlider
    brush_width_label: QLabel
    brush_width_slider: QSlider
    save_btn: QPushButton
    load_params_btn: QPushButton
    save_params_btn: QPushButton

    def __init__(
        self,
        current_fps: int = 30,
        current_frames: int = 150,
        grid_size_x: int = 25,
        grid_size_y: int = 35,
        x_limit: float = 5.0,
        y_limit: float = 5.0,
        initial_potential: Optional[np.ndarray] = None,
        initial_r0: Optional[np.ndarray] = None,
        initial_k0: Optional[np.ndarray] = None,
        initial_sigma: Optional[np.ndarray] = None,
        initial_mass: float = 1.0,
        initial_method: str = "Constant",
        initial_delta_t: float = 0.002,
        initial_steps_per_frame: int = 30,
        initial_wall_height: float = 50.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the drawing canvas and internal state.

        Args:
            current_fps (int): Currently active FPS configuration limit.
            current_frames (int): Currently active frame buffer duration limit.
            grid_size_x (int): Horizontal resolution of the simulation grid.
            grid_size_y (int): Vertical resolution of the simulation grid.
            x_limit (float): Maximum physical coordinate in X.
            y_limit (float): Maximum physical coordinate in Y.
            initial_potential (Optional[np.ndarray]): Previously saved potential matrix to restore.
            initial_r0 (Optional[np.ndarray]): Previously saved initial position vector.
            initial_k0 (Optional[np.ndarray]): Previously saved initial momentum vector.
            initial_sigma (Optional[np.ndarray]): Previously saved covariance matrix.
            initial_mass (float): Previously saved particle mass.
            initial_method (str): Previously selected simulation method.
            initial_delta_t (float): Previously saved time step size.
            initial_steps_per_frame (int): Previously saved physics sub-steps per animation frame.,
            initial_wall_height (float): Previously saved wall height.
            parent (Optional[QWidget]): Unused; transient parent is set by MainWindow
                after the window is shown.
        """
        super().__init__(None)

        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("Simulation Setup: Potential & Wavepacket")

        # Enforce minimum boundary constraints and add native OS window buttons (Minimize/Maximize)
        self.setMinimumSize(800, 800)
        self.resize(950, 900)

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        # Capture initial physics environment properties
        self.current_fps = current_fps
        self.current_frames = current_frames
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.x_limit = x_limit
        self.y_limit = y_limit

        self.initial_sigma = initial_sigma
        self.initial_mass = initial_mass
        self.initial_method = initial_method
        self.initial_delta_t = initial_delta_t
        self.initial_steps_per_frame = initial_steps_per_frame
        self.initial_wall_height = initial_wall_height

        # Canvas image is held at the native physics grid resolution; the widget is
        # purely an upscaled view of those exact (grid_size_x x grid_size_y) pixels.
        self.canvas = CanvasWidget(grid_size_x, grid_size_y)
        self.canvas_container = AspectRatioContainer(
            self.canvas, grid_size_y / grid_size_x
        )

        # Restore past matrix configurations if the user re-enters setup during runtime
        if initial_potential is not None:
            self._restore_canvas(initial_potential)

        self._setup_ui()

        # Restore the wavepacket directly in grid coordinates (no canvas-pixel detour),
        # so opening + saving the dialog untouched is a numerical no-op.
        if initial_r0 is not None and initial_k0 is not None:
            rx_grid = float(initial_r0[0])
            # Physics Y is bottom-up; Qt grid Y is top-down. Flip to draw upright.
            ry_grid = float(self.grid_size_y) - float(initial_r0[1])
            self.canvas.r0_grid = QPointF(rx_grid, ry_grid)

            kx_grid = float(initial_k0[0]) / K_GRID_FACTOR
            # Flip Y to match the top-down canvas convention
            ky_grid = -float(initial_k0[1]) / K_GRID_FACTOR
            self.canvas.k0_tip_grid = QPointF(rx_grid + kx_grid, ry_grid + ky_grid)

    def _restore_canvas(self, potential_array: np.ndarray) -> None:
        """
        Translates a raw numerical floating-point backend potential matrix back into
        the underlying canvas image. The image is built at native grid resolution
        and no spatial scaling is performed, so reopening the dialog reproduces the
        exact same pixels that were saved.

        Args:
            potential_array (np.ndarray): 2D array (size_x, size_y) representing the
                saved potential landscape.
        """
        # potential_array has shape (size_x, size_y) with row 0 at the canvas top
        # (see save_and_close); transposing yields (size_y, size_x) suitable for
        # QImage's (height, width) row-major layout.
        arr = 255.0 - (potential_array.T / self.initial_wall_height * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        height, width = arr.shape

        # Bring the image to the same grid resolution as the array before swapping in.
        if width != self.canvas.grid_size_x or height != self.canvas.grid_size_y:
            self.canvas.set_grid_size(width, height)

        gray_bytes = arr.tobytes()
        gray_img = QImage(
            gray_bytes,
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        ).copy()
        self.canvas.set_image(gray_img.convertToFormat(QImage.Format.Format_ARGB32))

    def _setup_ui(self) -> None:
        """
        Sets up the radio buttons, input fields, and layouts for the canvas dialog.
        Builds the visual interface layout dynamically via automated Qt structural managers.
        """
        layout = QVBoxLayout(self)

        # Simulation Method Layout
        sim_layout = QHBoxLayout()
        self.simulation_menu = QComboBox()
        self.simulation_menu.addItem("Crank-Nicolson")
        self.simulation_menu.addItem("SSFM")
        self.simulation_menu.addItem("Constant")
        self.simulation_menu.setCurrentText(self.initial_method)
        self.simulation_menu.currentTextChanged.connect(self.simulation_changed.emit)
        sim_layout.addWidget(QLabel("Simulation Method:"))
        sim_layout.addWidget(self.simulation_menu)
        sim_layout.addStretch()
        layout.addLayout(sim_layout)

        # UI/Video Parameters Layout
        sim_params_layout = QHBoxLayout()

        sim_params_layout.addWidget(QLabel("UI FPS:"))
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 120)
        self.fps_input.setValue(self.current_fps)
        sim_params_layout.addWidget(self.fps_input)
        sim_params_layout.addStretch()

        sim_params_layout.addWidget(QLabel("Total Frames:"))
        self.frames_input = QSpinBox()
        self.frames_input.setRange(10, 10000)
        self.frames_input.setValue(self.current_frames)
        sim_params_layout.addWidget(self.frames_input)
        sim_params_layout.addStretch()

        sim_params_layout.addWidget(QLabel("Grid X:"))
        self.size_x_input = QSpinBox()
        self.size_x_input.setRange(10, 1000)
        self.size_x_input.setValue(self.grid_size_x)
        self.size_x_input.valueChanged.connect(lambda _: self.check_memory_limit())
        sim_params_layout.addWidget(self.size_x_input)
        sim_params_layout.addStretch()

        sim_params_layout.addWidget(QLabel("Grid Y:"))
        self.size_y_input = QSpinBox()
        self.size_y_input.setRange(10, 1000)
        self.size_y_input.setValue(self.grid_size_y)
        self.size_y_input.valueChanged.connect(lambda _: self.check_memory_limit())
        sim_params_layout.addWidget(self.size_y_input)
        sim_params_layout.addStretch()

        self.update_grid_btn = QPushButton("Snap Aspect Ratio")
        self.update_grid_btn.clicked.connect(self.update_canvas_size)
        sim_params_layout.addWidget(self.update_grid_btn)

        sim_params_layout.addStretch()
        layout.addLayout(sim_params_layout)

        # Physics Parameters Layout
        physics_layout = QHBoxLayout()
        physics_layout.addWidget(QLabel("\u0394t (Time Step):"))
        self.delta_t_input = QDoubleSpinBox()
        self.delta_t_input.setDecimals(5)
        self.delta_t_input.setRange(0.00001, 1.0)
        self.delta_t_input.setSingleStep(0.001)
        self.delta_t_input.setValue(self.initial_delta_t)
        physics_layout.addWidget(self.delta_t_input)
        physics_layout.addStretch()

        physics_layout.addWidget(QLabel("Steps per Frame (\u0394n):"))
        self.steps_per_frame_input = QSpinBox()
        self.steps_per_frame_input.setRange(1, 2000)
        self.steps_per_frame_input.setValue(self.initial_steps_per_frame)
        physics_layout.addWidget(self.steps_per_frame_input)
        physics_layout.addStretch()

        physics_layout.addWidget(QLabel("Wall Height:"))
        self.wall_height_input = QDoubleSpinBox()
        self.wall_height_input.setRange(1.0, 1000000.0)
        self.wall_height_input.setSingleStep(10.0)
        self.wall_height_input.setValue(self.initial_wall_height)
        physics_layout.addWidget(self.wall_height_input)
        physics_layout.addStretch()

        layout.addLayout(physics_layout)

        # Preset Potential Selection Layout
        preset_layout = QHBoxLayout()
        self.preset_menu = QComboBox()
        self.preset_menu.addItem("Custom / Clear")
        self.preset_menu.addItem("Gaussian Bump")
        self.preset_menu.addItem("Harmonic Oscillator")
        self.preset_menu.addItem("W-shape")
        self.preset_menu.addItem("Matryoshka")

        self.preset_menu.textActivated.connect(self.load_preset_potential)
        preset_layout.addWidget(QLabel("Preset Potential:"))
        preset_layout.addWidget(self.preset_menu)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Mode Selection Layout
        mode_layout = QHBoxLayout()
        self.radio_brush = QRadioButton("Brush Potential")
        self.radio_brush.setChecked(True)
        self.radio_eraser = QRadioButton("Erase Potential")
        self.radio_wave = QRadioButton("Set Wavepacket")

        self.radio_brush.toggled.connect(self.update_mode)
        self.radio_eraser.toggled.connect(self.update_mode)
        self.radio_wave.toggled.connect(self.update_mode)

        mode_layout.addWidget(self.radio_brush)
        mode_layout.addWidget(self.radio_eraser)
        mode_layout.addWidget(self.radio_wave)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Wavepacket Parameters (Sigma Matrix & Mass)
        params_layout = QHBoxLayout()

        # Sigma xx with physical units
        params_layout.addWidget(QLabel("s<sub>xx</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_xx_input = QDoubleSpinBox()
        self.sig_xx_input.setRange(0.1, 100.0)
        self.sig_xx_input.setValue(15.0)
        self.sig_xx_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xx_input)
        params_layout.addStretch()

        # Sigma xy with physical units
        params_layout.addWidget(QLabel("s<sub>xy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_xy_input = QDoubleSpinBox()
        self.sig_xy_input.setRange(-100.0, 100.0)
        self.sig_xy_input.setValue(0.0)
        self.sig_xy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xy_input)
        params_layout.addStretch()

        # Sigma yy with physical units
        params_layout.addWidget(QLabel("s<sub>yy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_yy_input = QDoubleSpinBox()
        self.sig_yy_input.setRange(0.1, 100.0)
        self.sig_yy_input.setValue(15.0)
        self.sig_yy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_yy_input)
        params_layout.addStretch()

        # Filling the fields with initial matrix values if provided
        if self.initial_sigma is not None:
            self.sig_xx_input.setValue(float(self.initial_sigma[0, 0]))
            self.sig_xy_input.setValue(float(self.initial_sigma[0, 1]))
            self.sig_yy_input.setValue(float(self.initial_sigma[1, 1]))

        # Mass with physical units
        params_layout.addWidget(QLabel("m [m<sub>e</sub>]:"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.01, 100.0)
        self.mass_input.setValue(self.initial_mass)
        self.mass_input.setSingleStep(0.1)
        params_layout.addWidget(self.mass_input)
        params_layout.addStretch()

        layout.addLayout(params_layout)

        # Centered Interactivity Row containing the dynamically constrained Canvas
        canvas_area = QHBoxLayout()
        canvas_area.addWidget(self.canvas_container, stretch=1)
        canvas_area.addStretch()

        # Build the vertical slider layout
        slider_layout = QHBoxLayout()

        self.brush_strength_label = QLabel("Brush\nStrength: 15")
        self.brush_strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Brush width is measured in grid cells (matching the underlying image).
        self.brush_width_label = QLabel("Brush\nWidth: 3")
        self.brush_width_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.brush_strength_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_strength_slider.setRange(1, 100)
        self.brush_strength_slider.setValue(15)

        self.brush_width_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_width_slider.setRange(1, 30)
        self.brush_width_slider.setValue(3)

        def set_strength(v):
            self.brush_strength_label.setText(f"Brush\nStrength: {v}")
            self.canvas.brush_strength = v

        def set_width(v):
            self.brush_width_label.setText(f"Brush\nWidth: {v}")
            self.canvas.brush_width = v

        self.brush_strength_slider.valueChanged.connect(set_strength)
        self.brush_width_slider.valueChanged.connect(set_width)

        slider_layout.addWidget(self.brush_strength_label)
        slider_layout.addWidget(self.brush_strength_slider)

        slider_layout.addSpacing(40)

        slider_layout.addWidget(self.brush_width_slider)
        slider_layout.addWidget(self.brush_width_label)

        canvas_area.addLayout(slider_layout)

        layout.addLayout(canvas_area, stretch=1)

        # Action Controls
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Potential")
        clear_btn.clicked.connect(self.clear_canvas)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("Save && Update Simulation")
        self.save_btn.clicked.connect(self.save_and_close)

        self.load_params_btn = QPushButton("Load from JSON")
        self.load_params_btn.clicked.connect(self.load_params_from_file)

        self.save_params_btn = QPushButton("Save to JSON")
        self.save_params_btn.clicked.connect(self.save_params_to_file)

        controls.addWidget(self.load_params_btn)
        controls.addWidget(self.save_params_btn)
        controls.addStretch()
        controls.addWidget(clear_btn)
        controls.addWidget(cancel_btn)
        controls.addWidget(self.save_btn)

        layout.addLayout(controls)

        # Setting bigger SpinBox size for better visibility
        for spinbox in self.findChildren((QSpinBox, QDoubleSpinBox)):
            spinbox.setMinimumWidth(100)

        # Enforce memory safety on initial setup
        self.check_memory_limit()

    def update_canvas_size(self) -> None:
        """
        Dynamically resizes the drawing canvas to a new grid resolution. The
        underlying QImage is rescaled once here (a deliberate, user-initiated
        interpolation); the aspect-ratio container and stored wavepacket
        anchors are synchronised so on-screen content stays in place.
        """
        new_x = self.size_x_input.value()
        new_y = self.size_y_input.value()

        self.grid_size_x = new_x
        self.grid_size_y = new_y

        aspect_ratio = new_y / new_x
        self.canvas_container.set_aspect_ratio(aspect_ratio)
        self.canvas.set_grid_size(new_x, new_y)

    def check_memory_limit(self) -> None:
        """
        Calculates the maximum number of frames based on the grid size and available RAM.
        Lowers the frames input if it exceeds the calculated limit and notifies the user.
        """
        try:
            import psutil

            mem_available = psutil.virtual_memory().available
        except ImportError:
            # Fallback if psutil is not available, assume 16GB free memory
            mem_available = 16 * 1024 * 1024 * 1024

        # Reserve 2GB buffer for OS, cache
        safe_mem = max(0, mem_available - 2000 * 1024 * 1024)

        nx = self.size_x_input.value()
        ny = self.size_y_input.value()

        # Conservative estimation:
        # np.complex128 takes up 16 bytes
        bytes_per_frame = nx * ny * 16

        if bytes_per_frame == 0:
            return

        max_frames = int(safe_mem / bytes_per_frame)

        # Clamp to reasonable UI boundaries
        max_frames = min(max_frames, 10000)
        max_frames = max(max_frames, 10)

        current_frames = self.frames_input.value()

        # Update the maximum limit of the spinbox
        self.frames_input.setMaximum(max_frames)

        # Apply the reduction and notify if the current frames exceed the new limit
        if current_frames > max_frames:
            self.frames_input.setValue(max_frames)
            QMessageBox.warning(
                self,
                "Memory Limit Reached",
                f"The grid size is too large for the current number of frames.\n\n"
                f"Based on available RAM, the maximum number of frames has been safely lowered to {max_frames}.",
            )

    def load_preset_potential(self, text: str) -> None:
        """
        Loads a predefined mathematical potential from the backend onto the canvas.
        Replaces the current drawing with the generated matrix and configures optimal wavepacket paths.
        """
        if text == "Custom / Clear":
            self.clear_canvas()
            return

        potential_matrix = None
        wall_val = self.wall_height_input.value()

        if text == "Gaussian Bump":
            # Position the peak right in the middle of the simulated grid space
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            # Symmetric covariance matrix creating a smooth circular hill obstacle
            sigma0 = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float64)
            pot = GaussianBumpPotential(
                self.grid_size_x, self.grid_size_y, r0=r0, V0=wall_val, sigma0=sigma0
            )
            potential_matrix = pot.matrix

        elif text == "Harmonic Oscillator":
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            max_dist_sq = r0[0] ** 2 + r0[1] ** 2
            k = (wall_val * 2) / max_dist_sq if max_dist_sq > 0 else 1.0
            pot = HarmonicPotential(self.grid_size_x, self.grid_size_y, k=k, r0=r0)
            potential_matrix = pot.matrix

        elif text == "W-shape":
            # Custom W-shaped potential in the middle of the grid space
            w_size_x = self.grid_size_x // 2
            w_size_y = self.grid_size_y // 2

            pot = WShaped(w_size_x, w_size_y, thickness=3, wall_value=wall_val)
            w_matrix = pot.matrix

            pos_x = (self.grid_size_x - w_size_x) // 2
            pos_y = (self.grid_size_y - w_size_y) // 2

            zero_pot = np.zeros((self.grid_size_x, self.grid_size_y))
            zero_pot[pos_x : pos_x + w_size_x, pos_y : pos_y + w_size_y] = w_matrix
            potential_matrix = zero_pot

            # Drop the wavepacket near the top centre of the well with downward momentum.
            rx_grid = self.grid_size_x * 0.5
            ry_grid = self.grid_size_y * 0.2
            self.canvas.r0_grid = QPointF(rx_grid, ry_grid)
            # Arrow tip 20% of the canvas height below the anchor (positive Qt-Y =
            # negative physics-Y momentum after the save_and_close Y-flip).
            self.canvas.k0_tip_grid = QPointF(rx_grid, ry_grid + self.grid_size_y * 0.2)
            self.sig_xx_input.setValue(4.0)
            self.sig_yy_input.setValue(4.0)

        elif text == "Matryoshka":
            # Custom embedded potential with a central well and surrounding barriers
            # Calculating the inner potential from previous potential data
            scaled_img = self.canvas.image.scaled(
                self.grid_size_x // 2,
                self.grid_size_y // 2,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            gray_img = scaled_img.convertToFormat(QImage.Format.Format_Grayscale8)
            width, height = gray_img.width(), gray_img.height()
            bpl = gray_img.bytesPerLine()
            buffer = gray_img.constBits().asarray(height * bpl)

            arr = (
                np.frombuffer(bytes(buffer), dtype=np.uint8)
                .reshape((height, bpl))
                .copy()
            )
            arr = arr[:, :width]

            inner_matrix = ((255 - arr) / 255.0 * wall_val).T

            inner_matrix[0, :] = wall_val
            inner_matrix[-1, :] = wall_val
            inner_matrix[:, 0] = wall_val
            inner_matrix[:, -1] = wall_val

            inner_pot_obj = Potential(inner_matrix)

            pos_x = self.grid_size_x // 4
            pos_y = self.grid_size_y // 4

            pot = EmbeddedPotential(
                self.grid_size_x, self.grid_size_y, pos_x, pos_y, inner_pot_obj
            )
            potential_matrix = pot.matrix

        if potential_matrix is not None:
            # Leverage the existing canvas reconstruction pipeline
            self._restore_canvas(potential_matrix)
            self.update()

    def update_mode(self) -> None:
        """
        Updates the drawing state based on the current radio button selection.
        """
        if self.radio_brush.isChecked():
            self.canvas.mode = "brush"
        elif self.radio_eraser.isChecked():
            self.canvas.mode = "eraser"
        else:
            self.canvas.mode = "wavepacket"

    def clear_canvas(self) -> None:
        """
        Clears the drawn potential to a blank white canvas and resets the preset dropdown.
        """
        self.preset_menu.setCurrentText("Custom / Clear")
        self.canvas.image.fill(Qt.GlobalColor.white)
        self.canvas.update()

    def save_and_close(self) -> None:
        """
        Parses canvas drawing and physics inputs, emits them, and closes the dialog.
        """
        self.save_btn.setText("Loading...")
        self.save_btn.setEnabled(False)
        QApplication.processEvents()

        # Capture the newly desired grid sizes from user input
        new_size_x = self.size_x_input.value()
        new_size_y = self.size_y_input.value()
        wall_height = self.wall_height_input.value()

        # 1. Process Potential Matrix
        # The canvas image already lives at its current grid resolution. We only
        # interpolate here if the user explicitly changed the grid size input
        # without pressing "Snap Aspect Ratio" first.
        if (
            self.canvas.grid_size_x != new_size_x
            or self.canvas.grid_size_y != new_size_y
        ):
            img_at_grid = self.canvas.image.scaled(
                new_size_x,
                new_size_y,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            img_at_grid = self.canvas.image

        gray_img = img_at_grid.convertToFormat(QImage.Format.Format_Grayscale8)
        width, height = gray_img.width(), gray_img.height()
        bpl = gray_img.bytesPerLine()
        buffer = gray_img.constBits().asarray(height * bpl)

        arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * wall_height
        potential = potential.T

        # 2. Process Wavepacket Parameters in grid coordinates (no canvas-pixel detour)
        if self.canvas.r0_grid is not None and self.canvas.k0_tip_grid is not None:
            # Match the wavepacket to the requested grid resolution if it changed.
            scale_x = new_size_x / self.canvas.grid_size_x
            scale_y = new_size_y / self.canvas.grid_size_y

            r0x_g = self.canvas.r0_grid.x() * scale_x
            r0y_g = self.canvas.r0_grid.y() * scale_y
            k0x_g = self.canvas.k0_tip_grid.x() * scale_x
            k0y_g = self.canvas.k0_tip_grid.y() * scale_y

            rx = float(np.clip(r0x_g, 0.0, float(new_size_x - 1)))
            # Flip from top-down (Qt) back to bottom-up (physics)
            ry = float(np.clip(new_size_y - r0y_g, 0.0, float(new_size_y - 1)))
            r0 = np.array([rx, ry], dtype=np.float64)

            kx = (k0x_g - r0x_g) * K_GRID_FACTOR
            ky = -(k0y_g - r0y_g) * K_GRID_FACTOR
            k0 = np.array([kx, ky], dtype=np.float64)
        else:
            r0 = np.array([0.0, 0.0], dtype=np.float64)
            k0 = np.array([0.0, 0.0], dtype=np.float64)

        # 3. Process Sigma Matrix and Mass
        sig_xx = self.sig_xx_input.value()
        sig_xy = self.sig_xy_input.value()
        sig_yy = self.sig_yy_input.value()

        # Build the symmetric 2x2 covariance matrix
        sigma_matrix = np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])

        mass = self.mass_input.value()
        fps = self.fps_input.value()
        frames = self.frames_input.value()

        delta_t = self.delta_t_input.value()
        steps_per_frame = self.steps_per_frame_input.value()

        # Emit all parameters to the main window
        self.setup_saved.emit(
            potential,
            r0,
            k0,
            sigma_matrix,
            mass,
            fps,
            frames,
            new_size_x,
            new_size_y,
            delta_t,
            steps_per_frame,
            wall_height,
        )
        self.accept()

    def load_params_from_file(self) -> None:
        """
        Loads simulation parameters from a JSON file and updates the UI.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation Parameters", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        p = Params()
        p.read(file_path)

        # Update pure numeric fields
        self.size_x_input.setValue(p.size_x)
        self.size_y_input.setValue(p.size_y)
        self.mass_input.setValue(p.mass)
        self.frames_input.setValue(p.updates_max)

        self.delta_t_input.setValue(p.delta_t)
        self.steps_per_frame_input.setValue(p.delta_n)
        self.wall_height_input.setValue(p.well_height)

        # Update Sigma matrix inputs
        self.sig_xx_input.setValue(p.sigma0[0][0])
        self.sig_xy_input.setValue(p.sigma0[0][1])
        self.sig_yy_input.setValue(p.sigma0[1][1])

        # Map SolverType back to dropdown text
        if p.solver == SolverType.CN:
            self.simulation_menu.setCurrentText("Crank-Nicolson")
        elif p.solver == SolverType.SSFM:
            self.simulation_menu.setCurrentText("SSFM")
        

        # Map WellType back to dropdown presets
        if p.potential_matrix is not None:
            # Custom matrix loading
            self.preset_menu.setCurrentText("Custom / Clear")
            self._restore_canvas(p.potential_matrix)
        elif p.well_type == WellType.W_SHAPED:
            self.preset_menu.setCurrentText("W-shape")
            self.load_preset_potential("W-shape")
        elif p.well_type == WellType.MATRYOSHKA:
            self.preset_menu.setCurrentText("Matryoshka")
            self.load_preset_potential("Matryoshka")
        else:
            self.preset_menu.setCurrentText("Custom / Clear")
            self.load_preset_potential("Custom / Clear")

        self.update_canvas_size()

        rx_grid = float(p.r0[0])
        # Conversion from bottom-to-top (physics) to top-to-bottom (Qt grid) Y
        ry_grid = float(self.grid_size_y) - float(p.r0[1])
        self.canvas.r0_grid = QPointF(rx_grid, ry_grid)

        kx_grid = float(p.k0[0]) / K_GRID_FACTOR
        # Conversion from bottom-to-top (physics) to top-to-bottom (Qt grid) Y
        ky_grid = -float(p.k0[1]) / K_GRID_FACTOR
        self.canvas.k0_tip_grid = QPointF(rx_grid + kx_grid, ry_grid + ky_grid)

        self.canvas.update()

    def save_params_to_file(self) -> None:
        """
        Saves simulation parameters to a JSON file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Parameters", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        if file_path and not file_path.endswith(".json"):
            file_path += ".json"

        p = Params()
        p.size_x = self.size_x_input.value()
        p.size_y = self.size_y_input.value()
        p.mass = self.mass_input.value()
        p.updates_max = self.frames_input.value()
        p.delta_t = self.delta_t_input.value()
        p.delta_n = self.steps_per_frame_input.value()
        p.well_height = self.wall_height_input.value()

        p.sigma0 = np.array(
            [
                [self.sig_xx_input.value(), self.sig_xy_input.value()],
                [self.sig_xy_input.value(), self.sig_yy_input.value()],
            ],
            dtype=np.float64,
        )

        solver_text = self.simulation_menu.currentText()
        if solver_text == "Crank-Nicolson":
            p.solver = SolverType.CN
        elif solver_text == "SSFM" or solver_text == "Constant":
            p.solver = SolverType.SSFM

        preset_text = self.preset_menu.currentText()
        if preset_text == "W-shape":
            p.well_type = WellType.W_SHAPED
        elif preset_text == "Matryoshka":
            p.well_type = WellType.MATRYOSHKA
        else:
            p.well_type = WellType.INFINITE_WELL

        if self.canvas.r0_grid is not None and self.canvas.k0_tip_grid is not None:
            # Match the wavepacket to the requested grid resolution if it changed.
            scale_x = p.size_x / self.canvas.grid_size_x
            scale_y = p.size_y / self.canvas.grid_size_y

            r0x_g = self.canvas.r0_grid.x() * scale_x
            r0y_g = self.canvas.r0_grid.y() * scale_y
            k0x_g = self.canvas.k0_tip_grid.x() * scale_x
            k0y_g = self.canvas.k0_tip_grid.y() * scale_y

            rx = int(np.clip(r0x_g, 0, p.size_x - 1))
            # Flip from top-down (Qt) back to bottom-up (physics) and snap to int.
            ry = int(np.clip(p.size_y - r0y_g, 0, p.size_y - 1))
            p.r0 = (rx, ry)

            kx = float((k0x_g - r0x_g) * K_GRID_FACTOR)
            ky = float(-(k0y_g - r0y_g) * K_GRID_FACTOR)
            p.k0 = np.array([kx, ky], dtype=np.float64)
        else:
            p.r0 = (p.size_x // 2, p.size_y // 2)
            p.k0 = np.array([0.0, 0.0], dtype=np.float64)
        

        img_at_grid = self.canvas.image

        gray_img = img_at_grid.convertToFormat(QImage.Format.Format_Grayscale8)
        width, height = gray_img.width(), gray_img.height()
        bpl = gray_img.bytesPerLine()
        buffer = gray_img.constBits().asarray(height * bpl)

        arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * p.well_height
        
        p.potential_matrix = potential.T

        p.write(file_path)
