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
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage
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
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
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

        # Establish base starting dimensions for the canvas to present a larger drawing space
        base_width = 800
        base_height = int(base_width * (grid_size_y / grid_size_x))

        # Instantiate our modular, isolated interactive sketch canvas component
        self.canvas = CanvasWidget(base_width, base_height)
        # Wrap it in a protective container that enforces the physics matrix ratio dynamically
        self.canvas_container = AspectRatioContainer(
            self.canvas, base_height / base_width
        )

        # Restore past matrix configurations if the user re-enters setup during runtime
        if initial_potential is not None:
            self._restore_canvas(initial_potential)

        self._setup_ui()

        # Re-map numerical position states back into canvas pixel tags for visual continuous tracking
        if initial_r0 is not None and initial_k0 is not None:
            rx_px = int((initial_r0[0] / self.grid_size_x) * self.canvas.width())
            ry_px = int(
                (1.0 - (initial_r0[1] / self.grid_size_y)) * self.canvas.height()
            )
            self.canvas.r0_px = QPoint(rx_px, ry_px)

            kx_px = int((initial_k0[0] / 0.1) + rx_px)
            # Conversion from bottom-to-top top-to-bottom coordinate system
            ky_px = int((-initial_k0[1] / 0.1) + ry_px)
            self.canvas.k0_tip_px = QPoint(kx_px, ky_px)

    def _restore_canvas(self, potential_array: np.ndarray) -> None:
        """
        Translates a raw numerical floating-point backend potential matrix back into
        viewable canvas pixel values, rebuilding the visual environment layer.
        Reverses the calculations performed during save_and_close to reconstruct grayscale pixels.

        Args:
            potential_array (np.ndarray): 2D array representing the saved potential landscape.
        """
        arr = 255 - (potential_array.T / self.initial_wall_height * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.int32)

        height, width = arr.shape
        temp_img = QImage(width, height, QImage.Format.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                v = int(arr[y, x])
                temp_img.setPixelColor(x, y, QColor(v, v, v))

        restored_img = temp_img.scaled(
            self.canvas.width(),
            self.canvas.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.set_image(restored_img)

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

        sim_params_layout.addWidget(QLabel("Total Frames:"))
        self.frames_input = QSpinBox()
        self.frames_input.setRange(10, 10000)
        self.frames_input.setValue(self.current_frames)
        sim_params_layout.addWidget(self.frames_input)

        sim_params_layout.addWidget(QLabel("Grid X:"))
        self.size_x_input = QSpinBox()
        self.size_x_input.setRange(10, 1000)
        self.size_x_input.setValue(self.grid_size_x)
        self.size_x_input.valueChanged.connect(lambda _: self.check_memory_limit())
        sim_params_layout.addWidget(self.size_x_input)

        sim_params_layout.addWidget(QLabel("Grid Y:"))
        self.size_y_input = QSpinBox()
        self.size_y_input.setRange(10, 1000)
        self.size_y_input.setValue(self.grid_size_y)
        self.size_y_input.valueChanged.connect(lambda _: self.check_memory_limit())
        sim_params_layout.addWidget(self.size_y_input)

        # Dynamic Grid Resize Button
        self.update_grid_btn = QPushButton("Snap Aspect Ratio")
        self.update_grid_btn.clicked.connect(self.update_canvas_size)
        sim_params_layout.addWidget(self.update_grid_btn)

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

        physics_layout.addWidget(QLabel("Steps per Frame (\u0394n):"))
        self.steps_per_frame_input = QSpinBox()
        self.steps_per_frame_input.setRange(1, 2000)
        self.steps_per_frame_input.setValue(self.initial_steps_per_frame)
        physics_layout.addWidget(self.steps_per_frame_input)

        physics_layout.addWidget(QLabel("Wall Height:"))
        self.wall_height_input = QDoubleSpinBox()
        self.wall_height_input.setRange(1.0, 1000000.0)
        self.wall_height_input.setSingleStep(10.0)
        self.wall_height_input.setValue(self.initial_wall_height)
        physics_layout.addWidget(self.wall_height_input)
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
        self.sig_xx_input.setRange(0.1, 20.0)
        self.sig_xx_input.setValue(1.0)
        self.sig_xx_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xx_input)

        # Sigma xy with physical units
        params_layout.addWidget(QLabel("s<sub>xy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_xy_input = QDoubleSpinBox()
        self.sig_xy_input.setRange(-10.0, 10.0)
        self.sig_xy_input.setValue(0.0)
        self.sig_xy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xy_input)

        # Sigma yy with physical units
        params_layout.addWidget(QLabel("s<sub>yy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_yy_input = QDoubleSpinBox()
        self.sig_yy_input.setRange(0.1, 20.0)
        self.sig_yy_input.setValue(1.0)
        self.sig_yy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_yy_input)

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

        layout.addLayout(params_layout)

        # Centered Interactivity Row containing the dynamically constrained Canvas
        canvas_area = QHBoxLayout()
        canvas_area.addWidget(self.canvas_container, stretch=1)
        canvas_area.addSpacing(20)

        # Build the vertical slider layout
        slider_layout = QHBoxLayout()

        self.brush_strength_label = QLabel("Brush\nStrength: 15")
        self.brush_strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.brush_width_label = QLabel("Brush\nWidth: 30")
        self.brush_width_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.brush_strength_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_strength_slider.setRange(1, 100)
        self.brush_strength_slider.setValue(15)

        self.brush_width_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_width_slider.setRange(10, 100)
        self.brush_width_slider.setValue(30)

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

        self.save_btn = QPushButton("Save & Update Simulation")
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

        # Enforce memory safety on initial setup
        self.check_memory_limit()

    def update_canvas_size(self) -> None:
        """
        Dynamically resizes the drawing canvas aspect bounds based on the specified grid resolution.
        Ensures the drawing aspect ratio immediately reflects the target physical simulation.
        """
        new_x = self.size_x_input.value()
        new_y = self.size_y_input.value()

        self.grid_size_x = new_x
        self.grid_size_y = new_y

        # Push the new ratio restriction down to the container (triggers auto-resize)
        aspect_ratio = new_y / new_x
        self.canvas_container.set_aspect_ratio(aspect_ratio)

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

            # Setting initial wavepacket position and momentum
            rx_px = int(self.canvas.width() * 0.5)
            ry_px = int(self.canvas.height() * 0.2)
            self.canvas.r0_px = QPoint(rx_px, ry_px)
            self.canvas.k0_tip_px = QPoint(rx_px, ry_px + 80)
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

            arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape(
                (height, bpl)
            ).copy()
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
        # Scale canvas directly to the newly selected grid size
        scaled_img = self.canvas.image.scaled(
            new_size_x,
            new_size_y,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        gray_img = scaled_img.convertToFormat(QImage.Format.Format_Grayscale8)
        width, height = gray_img.width(), gray_img.height()
        bpl = gray_img.bytesPerLine()
        buffer = gray_img.constBits().asarray(height * bpl)

        arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * wall_height
        potential = potential.T

        # 2. Process Wavepacket Parameters
        if self.canvas.r0_px is not None and self.canvas.k0_tip_px is not None:
            # Map Pixel X to a natural number [0, new_size_x - 1]
            rx_float = (self.canvas.r0_px.x() / self.canvas.width()) * new_size_x
            rx = int(np.clip(rx_float, 0, new_size_x - 1))

            # Map Pixel Y to a natural number [0, new_size_y - 1] (Inverting so 0 is at bottom)
            # Conversion from top-to-bottom to bottom-to-top coordinate system
            ry_float = (
                1.0 - (self.canvas.r0_px.y() / self.canvas.height())
            ) * new_size_y
            ry = int(np.clip(ry_float, 0, new_size_y - 1))

            r0 = np.array([rx, ry])

            # Map Pixel X and Y to momentum components
            kx = (self.canvas.k0_tip_px.x() - self.canvas.r0_px.x()) * 0.1
            # Conversion from top-to-bottom to bottom-to-top coordinate system
            ky = -(self.canvas.k0_tip_px.y() - self.canvas.r0_px.y()) * 0.1
            k0 = np.array([kx, ky])
        else:
            r0 = np.array([0, 0])
            k0 = np.array([0.0, 0.0])

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
        if p.well_type == WellType.W_SHAPED:
            self.preset_menu.setCurrentText("W-shape")
            self.load_preset_potential("W-shape")
        elif p.well_type == WellType.MATRYOSHKA:
            self.preset_menu.setCurrentText("Matryoshka")
            self.load_preset_potential("Matryoshka")
        else:
            self.preset_menu.setCurrentText("Custom / Clear")
            self.load_preset_potential("Custom / Clear")

        self.update_canvas_size()

        rx_px = int((p.r0[0] / p.size_x) * self.canvas.width())
        # Conversion from bottom-to-top to top-to-bottom coordinate system
        ry_px = int((1.0 - (p.r0[1] / p.size_y)) * self.canvas.height())
        self.canvas.r0_px = QPoint(rx_px, ry_px)

        kx_px = int((p.k0[0] / 0.1) + rx_px)
        # Conversion from bottom-to-top to top-to-bottom coordinate system
        ky_px = int((-p.k0[1] / 0.1) + ry_px)
        self.canvas.k0_tip_px = QPoint(kx_px, ky_px)

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

        # Prawidłowy zapis fizyki
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

        if self.canvas.r0_px is not None and self.canvas.k0_tip_px is not None:
            rx_float = (self.canvas.r0_px.x() / self.canvas.width()) * p.size_x
            rx = int(np.clip(rx_float, 0, p.size_x - 1))

            ry_float = (1.0 - (self.canvas.r0_px.y() / self.canvas.height())) * p.size_y
            ry = int(np.clip(ry_float, 0, p.size_y - 1))
            p.r0 = (rx, ry)

            kx = float((self.canvas.k0_tip_px.x() - self.canvas.r0_px.x()) * 0.1)
            ky = float(-(self.canvas.k0_tip_px.y() - self.canvas.r0_px.y()) * 0.1)
            p.k0 = np.array([kx, ky], dtype=np.float64)
        else:
            p.r0 = (p.size_x // 2, p.size_y // 2)
            p.k0 = np.array([0.0, 0.0], dtype=np.float64)

        p.write(file_path)
