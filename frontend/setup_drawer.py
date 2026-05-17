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
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Native module import incorporating our custom isolated drawing canvas components
from .canvas_widget import CanvasWidget, AspectRatioContainer


class SetupDrawer(QDialog):
    """
    A comprehensive physics configuration dialogue interface linking user sketches to simulation parameters.
    
    This window encapsulates input form layout parameters (discretization grids, playback rates, particle masses)
    and binds them alongside an interactive canvas element. Upon closure via submission, it performs 
    coordinate mapping and grayscale matrix translation to dispatch structured raw simulation data fields 
    back to the primary 3D visualization window engine.
    """

    # Communication pipelines emitting state data packets to notify the primary window manager
    simulation_changed = pyqtSignal(str)
    setup_saved = pyqtSignal(
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int, int, int
    )

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
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the configuration dashboard dialog, restoring current parameters into editable UI forms.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Setup: Potential & Wavepacket")
        
        # Enforce elastic minimum boundary constraints and add native OS window buttons (Minimize/Maximize)
        self.setMinimumSize(800, 750)
        self.resize(1000, 900)  # Start with a nice, large window!

        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.WindowMinimizeButtonHint | 
            Qt.WindowType.WindowMaximizeButtonHint | 
            Qt.WindowType.WindowCloseButtonHint
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

        # Establish base starting dimensions for the canvas to present a larger drawing space
        base_width = 800
        base_height = int(base_width * (grid_size_y / grid_size_x))

        # Instantiate our modular, isolated interactive sketch canvas component
        self.canvas = CanvasWidget(base_width, base_height)
        # Wrap it in a protective container that enforces the physics matrix ratio dynamically
        self.canvas_container = AspectRatioContainer(self.canvas, base_height / base_width)

        # Restore past matrix configurations if the user re-enters setup during runtime
        if initial_potential is not None:
            self._restore_canvas(initial_potential)

        self._setup_ui()

        # Re-map numerical position states back into canvas pixel tags for visual continuous tracking
        if initial_r0 is not None and initial_k0 is not None:
            rx_px = int((initial_r0[0] / self.grid_size_x) * self.canvas.width())
            ry_px = int((1.0 - (initial_r0[1] / self.grid_size_y)) * self.canvas.height())
            self.canvas.r0_px = QPoint(rx_px, ry_px)

            kx_px = int((initial_k0[0] / 0.1) + rx_px)
            ky_px = int((-initial_k0[1] / 0.1) + ry_px)
            self.canvas.k0_tip_px = QPoint(kx_px, ky_px)

    def _restore_canvas(self, potential_array: np.ndarray) -> None:
        """
        Translates a raw numerical floating-point backend potential matrix back into 
        viewable canvas pixel values, rebuilding the visual environment layer.
        """
        arr = 255 - (potential_array.T / 50.0 * 255.0)
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
        Builds the visual interface layout, arranging input fields, dropdown menus, 
        and form buttons dynamically via automated Qt structural managers.
        """
        layout = QVBoxLayout(self)

        # Block 1: Numerical Solver Selection Header
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

        # Block 2: Spatial Discretization and Core Configuration Rows
        sim_params_layout = QHBoxLayout()

        sim_params_layout.addWidget(QLabel("FPS:"))
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 120)
        self.fps_input.setValue(self.current_fps)
        sim_params_layout.addWidget(self.fps_input)

        sim_params_layout.addWidget(QLabel("Frames:"))
        self.frames_input = QSpinBox()
        self.frames_input.setRange(10, 10000)
        self.frames_input.setValue(self.current_frames)
        sim_params_layout.addWidget(self.frames_input)

        sim_params_layout.addWidget(QLabel("Grid X:"))
        self.size_x_input = QSpinBox()
        self.size_x_input.setRange(10, 500)
        self.size_x_input.setValue(self.grid_size_x)
        sim_params_layout.addWidget(self.size_x_input)

        sim_params_layout.addWidget(QLabel("Grid Y:"))
        self.size_y_input = QSpinBox()
        self.size_y_input.setRange(10, 500)
        self.size_y_input.setValue(self.grid_size_y)
        sim_params_layout.addWidget(self.size_y_input)

        self.update_grid_btn = QPushButton("Snap Aspect Ratio")
        self.update_grid_btn.clicked.connect(self.update_canvas_size)
        sim_params_layout.addWidget(self.update_grid_btn)

        layout.addLayout(sim_params_layout)

        # Block 3: Pre-calculated Mathematical Presets Dropdown Module
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

        # Block 4: Interaction Mode Selection Radio Switches
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

        # Block 5: Quantum Mechanics Form Fields (Mass and Covariance Properties)
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("s<sub>xx</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_xx_input = QDoubleSpinBox()
        self.sig_xx_input.setRange(0.1, 20.0)
        self.sig_xx_input.setValue(1.0)
        self.sig_xx_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xx_input)

        params_layout.addWidget(QLabel("s<sub>xy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_xy_input = QDoubleSpinBox()
        self.sig_xy_input.setRange(-10.0, 10.0)
        self.sig_xy_input.setValue(0.0)
        self.sig_xy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xy_input)

        params_layout.addWidget(QLabel("s<sub>yy</sub> [a<sub>0</sub><sup>2</sup>]:"))
        self.sig_yy_input = QDoubleSpinBox()
        self.sig_yy_input.setRange(0.1, 20.0)
        self.sig_yy_input.setValue(1.0)
        self.sig_yy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_yy_input)

        if self.initial_sigma is not None:
            self.sig_xx_input.setValue(float(self.initial_sigma[0, 0]))
            self.sig_xy_input.setValue(float(self.initial_sigma[0, 1]))
            self.sig_yy_input.setValue(float(self.initial_sigma[1, 1]))

        params_layout.addWidget(QLabel("m [m<sub>e</sub>]:"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.01, 100.0)
        self.mass_input.setValue(self.initial_mass)
        self.mass_input.setSingleStep(0.1)
        params_layout.addWidget(self.mass_input)

        layout.addLayout(params_layout)

        # Block 6: Centered Interactivity Row containing the dynamically constrained Canvas
        canvas_area = QHBoxLayout()
        # The AspectRatioContainer automatically resizes the internal canvas!
        canvas_area.addWidget(self.canvas_container, stretch=1)
        canvas_area.addSpacing(20)

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

        # Map vertical layout slider triggers straight into properties on our CanvasWidget instance
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

        # By applying stretch=1 to the ENTIRE canvas area, we ensure it aggressively consumes all vertical space!
        layout.addLayout(canvas_area, stretch=1)

        # Block 7: Terminal Operation Control Row
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Potential")
        clear_btn.clicked.connect(self.clear_canvas)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("Save & Update Simulation")
        self.save_btn.clicked.connect(self.save_and_close)

        controls.addWidget(clear_btn)
        controls.addWidget(cancel_btn)
        controls.addWidget(self.save_btn)

        layout.addLayout(controls)

    def update_canvas_size(self) -> None:
        """
        Informs the dynamic aspect ratio wrapper to change its framing lock 
        based on newly provided physics grid resolutions.
        """
        new_x = self.size_x_input.value()
        new_y = self.size_y_input.value()

        self.grid_size_x = new_x
        self.grid_size_y = new_y

        # Push the new ratio restriction down to the container (triggers auto-resize)
        aspect_ratio = new_y / new_x
        self.canvas_container.set_aspect_ratio(aspect_ratio)

    def load_preset_potential(self, text: str) -> None:
        """
        Constructs specialized mathematical potential structures using parameters 
        derived from backend logic models and projects them onto the visual workspace layer.
        """
        if text == "Custom / Clear":
            self.clear_canvas()
            return

        potential_matrix = None

        if text == "Gaussian Bump":
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            V0 = 40.0
            sigma0 = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float64)
            pot = GaussianBumpPotential(
                self.grid_size_x, self.grid_size_y, r0=r0, V0=V0, sigma0=sigma0
            )
            potential_matrix = pot.matrix

        elif text == "Harmonic Oscillator":
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            max_dist_sq = r0[0] ** 2 + r0[1] ** 2
            k = 100.0 / max_dist_sq if max_dist_sq > 0 else 1.0
            pot = HarmonicPotential(self.grid_size_x, self.grid_size_y, k=k, r0=r0)
            potential_matrix = pot.matrix

        elif text == "W-shape":
            w_size_x = self.grid_size_x // 2
            w_size_y = self.grid_size_y // 2

            pot = WShaped(w_size_x, w_size_y, thickness=3, wall_value=50.0)
            w_matrix = pot.matrix.T

            pos_x = (self.grid_size_x - w_size_x) // 2
            pos_y = (self.grid_size_y - w_size_y) // 2

            zero_pot = np.zeros((self.grid_size_x, self.grid_size_y))
            zero_pot[pos_x : pos_x + w_size_x, pos_y : pos_y + w_size_y] = w_matrix
            potential_matrix = zero_pot

            # Optimize particle trajectory to fall cleanly into the W channel setup
            rx_px = int(self.canvas.width() * 0.5)
            ry_px = int(self.canvas.height() * 0.2)
            self.canvas.r0_px = QPoint(rx_px, ry_px)
            self.canvas.k0_tip_px = QPoint(rx_px, ry_px + 80)
            self.sig_xx_input.setValue(4.0)
            self.sig_yy_input.setValue(4.0)

        elif text == "Matryoshka":
            # Extract current visual sketches, downscale them, and wrap them in solid potential walls
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

            arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bpl)).copy()
            arr = arr[:, :width]

            inner_matrix = ((255 - arr) / 255.0 * 50).T
            inner_matrix[0, :] = 50.0
            inner_matrix[-1, :] = 50.0
            inner_matrix[:, 0] = 50.0
            inner_matrix[:, -1] = 50.0

            inner_pot_obj = Potential(inner_matrix)

            pos_x = self.grid_size_x // 4
            pos_y = self.grid_size_y // 4

            pot = EmbeddedPotential(
                self.grid_size_x, self.grid_size_y, pos_x, pos_y, inner_pot_obj
            )
            potential_matrix = pot.matrix

        if potential_matrix is not None:
            self._restore_canvas(potential_matrix)
            self.update()

    def update_mode(self) -> None:
        """Binds state alterations to the active tool layout radio choices."""
        if self.radio_brush.isChecked():
            self.canvas.mode = "brush"
        elif self.radio_eraser.isChecked():
            self.canvas.mode = "eraser"
        else:
            self.canvas.mode = "wavepacket"

    def clear_canvas(self) -> None:
        """Resets the custom sketch canvas back to an empty baseline."""
        self.preset_menu.setCurrentText("Custom / Clear")
        self.canvas.image.fill(Qt.GlobalColor.white)
        self.canvas.update()

    def save_and_close(self) -> None:
        """
        Compresses image arrays into a numerical potential grid, evaluates input 
        variables, emits a final physics configuration packet, and safely closes the dialog.
        """
        self.save_btn.setText("Loading...")
        self.save_btn.setEnabled(False)
        QApplication.processEvents()

        new_size_x = self.size_x_input.value()
        new_size_y = self.size_y_input.value()

        # Scale canvas pixels straight down into the exact physics discretization matrix size
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

        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * 50
        potential = potential.T

        # Translate interactive pixel coordinate pointers into raw physical state variables
        if self.canvas.r0_px and self.canvas.k0_tip_px:
            rx_float = (self.canvas.r0_px.x() / self.canvas.width()) * new_size_x
            rx = int(np.clip(rx_float, 0, new_size_x - 1))

            ry_float = (1.0 - (self.canvas.r0_px.y() / self.canvas.height())) * new_size_y
            ry = int(np.clip(ry_float, 0, new_size_y - 1))

            r0 = np.array([rx, ry])

            kx = (self.canvas.k0_tip_px.x() - self.canvas.r0_px.x()) * 0.1
            ky = -(self.canvas.k0_tip_px.y() - self.canvas.r0_px.y()) * 0.1
            k0 = np.array([kx, ky])
        else:
            r0 = np.array([0, 0])
            k0 = np.array([0.0, 0.0])

        sig_xx = self.sig_xx_input.value()
        sig_xy = self.sig_xy_input.value()
        sig_yy = self.sig_yy_input.value()

        sigma_matrix = np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])

        mass = self.mass_input.value()
        fps = self.fps_input.value()
        frames = self.frames_input.value()

        # Emit the fully assembled parameter packet straight out to the primary application thread
        self.setup_saved.emit(
            potential, r0, k0, sigma_matrix, mass, fps, frames, new_size_x, new_size_y
        )
        self.accept()