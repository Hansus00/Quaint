# ==============================================================================
# ### --- FILE frontend/setup_drawer.py --- ###
# ==============================================================================

from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
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
    QVBoxLayout,
    QWidget,
)

from backend.Potential import Potential, GaussianBumpPotential, HarmonicPotential, WShaped, EmbeddedPotential


class SetupDrawer(QDialog):
    """
    A 2D canvas dialog for drawing a potential field and setting wavepacket starting states.
    Emits a tuple upon saving containing: (potential_matrix, r0, k0, sigma_matrix, mass)
    """

    # -- Class Fields --
    simulation_changed = pyqtSignal(str)
    setup_saved = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)

    grid_size_x: int
    grid_size_y: int
    x_limit: float
    y_limit: float
    initial_sigma: Optional[np.ndarray]
    initial_mass: float
    initial_method: str
    initial_r0: Optional[np.ndarray]
    initial_k0: Optional[np.ndarray]
    canvas_width: int
    canvas_height: int
    image: QImage
    drawing_potential: bool
    mode: str
    last_point: QPoint
    r0_px: Optional[QPoint]
    k0_tip_px: Optional[QPoint]
    simulation_menu: QComboBox
    preset_menu: QComboBox
    radio_brush: QRadioButton
    radio_eraser: QRadioButton
    radio_wave: QRadioButton
    sig_xx_input: QDoubleSpinBox
    sig_xy_input: QDoubleSpinBox
    sig_yy_input: QDoubleSpinBox
    mass_input: QDoubleSpinBox
    brush_strength_label: QLabel
    brush_strength_slider: QSlider
    save_btn: QPushButton

    def __init__(
        self,
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
        Initializes the drawing canvas.

        Args:
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
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Setup: Potential & Wavepacket")

        self.grid_size_x: int = grid_size_x
        self.grid_size_y: int = grid_size_y
        self.x_limit: float = x_limit
        self.y_limit: float = y_limit

        self.initial_sigma = initial_sigma
        self.initial_mass = initial_mass
        self.initial_method = initial_method
        self.initial_r0 = initial_r0
        self.initial_k0 = initial_k0

        self.canvas_width: int = 400
        self.canvas_height: int = int(400 * (grid_size_y / grid_size_x))

        self.image = QImage(
            self.canvas_width, self.canvas_height, QImage.Format.Format_ARGB32
        )
        self.image.fill(Qt.GlobalColor.white)

        # Restoring the potential drawing on the canvas
        if initial_potential is not None:
            self._restore_canvas(initial_potential)

        # State Variables
        self.drawing_potential: bool = False
        self.mode: str = "brush"  # Options: "brush", "eraser", "wavepacket"
        self.last_point: QPoint = QPoint()

        # Wavepacket vector state (stored in pixel coordinates for the UI)
        self.r0_px: Optional[QPoint] = None
        self.k0_tip_px: Optional[QPoint] = None

        self._setup_ui()

    def _restore_canvas(self, potential_array: np.ndarray) -> None:
        """
        Restores the drawn image on the canvas from the raw potential matrix.
        Reverses the calculations performed during save_and_close to reconstruct grayscale pixels.

        Args:
            potential_array (np.ndarray): 2D array representing the saved potential landscape.
        """
        arr = 255 - (potential_array.T / 50.0 * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.int32)

        height, width = arr.shape
        temp_img = QImage(width, height, QImage.Format.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                v = int(arr[y, x])
                temp_img.setPixelColor(x, y, QColor(v, v, v))

        self.image = temp_img.scaled(
            self.canvas_width,
            self.canvas_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _setup_ui(self) -> None:
        """Sets up the radio buttons, input fields, and layouts for the canvas dialog."""
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

        # Preset Potential Selection Layout
        preset_layout = QHBoxLayout()
        self.preset_menu = QComboBox()
        self.preset_menu.addItem("Custom / Clear")
        self.preset_menu.addItem("Gaussian Bump")
        self.preset_menu.addItem("Harmonic Oscillator")
        self.preset_menu.addItem("W-shape")
        self.preset_menu.addItem("Embedded")
        self.preset_menu.currentTextChanged.connect(self.load_preset_potential)
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

        # Physics Parameters (Sigma Matrix & Mass)
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

        # Canvas area wrapper with Brush Slider on the right
        canvas_area = QHBoxLayout()
        # Add a fixed spacing exactly the width of the canvas to prevent overlaps
        canvas_area.addSpacing(self.canvas_width + 20)

        # Build the vertical slider layout
        slider_layout = QHBoxLayout()

        self.brush_strength_label = QLabel("Brush\nStrength: 15")
        self.brush_strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.brush_width_label = QLabel("Brush\nWidth: 15")
        self.brush_width_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.brush_strength_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_strength_slider.setRange(1, 100)
        self.brush_strength_slider.setValue(15)  # Default starting alpha
        self.brush_strength_slider.setMinimumHeight(self.canvas_height - 60)
        self.brush_strength_slider.valueChanged.connect(
            lambda v: self.brush_strength_label.setText(f"Brush\nStrength: {v}")
        )

        self.brush_width_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_width_slider.setRange(10, 100)
        self.brush_width_slider.setValue(30)  # Default starting alpha
        self.brush_width_slider.setMinimumHeight(self.canvas_height - 60)
        self.brush_width_slider.valueChanged.connect(
            lambda v: self.brush_width_label.setText(f"Brush\nWidth: {v}")
        )

        slider_layout.addWidget(self.brush_strength_label)
        slider_layout.addWidget(self.brush_strength_slider)
        slider_layout.addWidget(self.brush_width_slider)
        slider_layout.addWidget(self.brush_width_label)
        slider_layout.addStretch()

        canvas_area.addLayout(slider_layout)
        canvas_area.addStretch()
        
        layout.addLayout(canvas_area)

        # Canvas constraints (Increased height due to new preset row)
        self.setFixedSize(600, self.canvas_height + 200)

        # Restoring the positions of the wavepacket indicators on the canvas
        if self.initial_r0 is not None and self.initial_k0 is not None:
            rx_px = int((self.initial_r0[0] / self.grid_size_x) * self.canvas_width)
            ry_px = int((1.0 - (self.initial_r0[1] / self.grid_size_y)) * self.canvas_height)
            self.r0_px = QPoint(rx_px, ry_px)

            kx_px = int((self.initial_k0[0] / 0.1) + rx_px)
            ky_px = int((-self.initial_k0[1] / 0.1) + ry_px)
            self.k0_tip_px = QPoint(kx_px, ky_px)

        # Action Controls
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Potential")
        clear_btn.clicked.connect(self.clear_canvas)

        self.save_btn = QPushButton("Save & Update Simulation")
        self.save_btn.clicked.connect(self.save_and_close)

        controls.addWidget(clear_btn)
        controls.addWidget(self.save_btn)

        layout.addStretch()
        layout.addLayout(controls)

    def load_preset_potential(self, text: str) -> None:
        """
        Loads a predefined mathematical potential from the backend onto the canvas.
        Replaces the current drawing with the generated matrix.
        """
        if text == "Custom / Clear":
            self.clear_canvas()
            return

        potential_matrix = None

        if text == "Gaussian Bump":
            # Position the peak right in the middle of the simulated grid space
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            V0 = 40.0
            # Symmetric covariance matrix creating a smooth circular hill obstacle
            sigma0 = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float64)
            pot = GaussianBumpPotential(self.grid_size_x, self.grid_size_y, r0=r0, V0=V0, sigma0=sigma0)
            potential_matrix = pot.matrix

        elif text == "Harmonic Oscillator":
            # Calculate a spring constant k that naturally climbs to V~50 at the boundaries
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            max_dist_sq = r0[0] ** 2 + r0[1] ** 2
            k = 100.0 / max_dist_sq if max_dist_sq > 0 else 1.0
            pot = HarmonicPotential(self.grid_size_x, self.grid_size_y, k=k, r0=r0)
            potential_matrix = pot.matrix

        elif text == "W-shape":
            # Custom W-shaped potential in the middle of the grid space
            w_size_x = self.grid_size_x // 2
            w_size_y = self.grid_size_y // 2
            
            pot = WShaped(w_size_x, w_size_y, thickness=3, wall_value=50.0)
            
            w_matrix = pot.matrix.T
            
            pos_x = (self.grid_size_x - w_size_x) // 2
            pos_y = (self.grid_size_y - w_size_y) // 2
            
            zero_pot = np.zeros((self.grid_size_x, self.grid_size_y))
            
            zero_pot[pos_x:pos_x + w_size_x, pos_y:pos_y + w_size_y] = w_matrix
            potential_matrix = zero_pot

            # Setting initial wavepacket position and momentum
            rx_px = int(self.canvas_width * 0.5)
            ry_px = int(self.canvas_height * 0.2)
            self.r0_px = QPoint(rx_px, ry_px)
            self.k0_tip_px = QPoint(rx_px, ry_px + 80)
            self.sig_xx_input.setValue(4.0)
            self.sig_yy_input.setValue(4.0)
        
        elif text == "Embedded":
            # Custom Embedded potential with a central well and surrounding barriers

            # Calculating the inner potential from previous potential data
            scaled_img = self.image.scaled(
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

            # Setting boundaries
            inner_matrix[0, :] = 50.0
            inner_matrix[-1, :] = 50.0
            inner_matrix[:, 0] = 50.0
            inner_matrix[:, -1] = 50.0
            
            inner_pot_obj = Potential(inner_matrix)
            
            pos_x = self.grid_size_x // 4
            pos_y = self.grid_size_y // 4
            
            pot = EmbeddedPotential(
                self.grid_size_x, 
                self.grid_size_y, 
                pos_x, 
                pos_y, 
                inner_pot_obj
            )
            potential_matrix = pot.matrix

        if potential_matrix is not None:
            # Leverage the existing canvas reconstruction pipeline
            self._restore_canvas(potential_matrix)
            self.update()

    def update_mode(self) -> None:
        """Updates the drawing state based on radio button selection."""
        if self.radio_brush.isChecked():
            self.mode = "brush"
        elif self.radio_eraser.isChecked():
            self.mode = "eraser"
        else:
            self.mode = "wavepacket"

    def paintEvent(self, event) -> None:
        """Handles the rendering of the canvas and overlay objects like the wave vector."""
        canvas_painter = QPainter(self)

        # Offset Y to account for top UI controls (adjusted for new row layout)
        offset_y = 135
        canvas_painter.translate(0, offset_y)
        canvas_painter.drawImage(0, 0, self.image)

        if self.r0_px and self.k0_tip_px:
            pen = QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine)
            canvas_painter.setPen(pen)
            canvas_painter.setBrush(Qt.GlobalColor.red)
            canvas_painter.drawLine(self.r0_px, self.k0_tip_px)
            canvas_painter.drawEllipse(self.r0_px, 4, 4)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint() - QPoint(0, 135)

        # Stop interactions if clicked outside the canvas bounds
        if pos.x() < 0 or pos.x() > self.canvas_width or pos.y() < 0 or pos.y() > self.canvas_height:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser"):
                self.drawing_potential = True
                self.last_point = pos
            elif self.mode == "wavepacket":
                self.r0_px = pos
                self.k0_tip_px = pos
                self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint() - QPoint(0, 135)

        # Stop interactions if cursor moves outside the canvas bounds
        if pos.x() < 0 or pos.x() > self.canvas_width or pos.y() < 0 or pos.y() > self.canvas_height:
            return

        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser") and self.drawing_potential:
                painter = QPainter(self.image)
                width = self.brush_width_slider.value()
                # Brush adds dark semi-transparent strokes; Eraser overwrites with solid white
                if self.mode == "brush":
                    strength = self.brush_strength_slider.value()
                    color = QColor(0, 0, 0, strength)
                else:
                    color = QColor(255, 255, 255, 255)
                    
                pen = QPen(
                    color,
                    width,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(pen)
                painter.drawLine(self.last_point, pos)
                self.last_point = pos
                self.update()

            elif self.mode == "wavepacket":
                self.k0_tip_px = pos
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing_potential = False

    def clear_canvas(self) -> None:
        """Clears the drawn potential to a blank white canvas and resets the preset dropdown."""
        self.preset_menu.setCurrentText("Custom / Clear")
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def save_and_close(self) -> None:
        """
        Parses canvas drawing and physics inputs, emits them, and closes the dialog.
        """
        self.save_btn.setText("Loading...")
        self.save_btn.setEnabled(False) 
        QApplication.processEvents()

        # 1. Process Potential Matrix
        scaled_img = self.image.scaled(
            self.grid_size_x,
            self.grid_size_y,
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

        # 2. Process Wavepacket Parameters
        if self.r0_px and self.k0_tip_px:
            # Map Pixel X to a natural number [0, grid_size_x - 1]
            rx_float = (self.r0_px.x() / self.canvas_width) * self.grid_size_x
            rx = int(np.clip(rx_float, 0, self.grid_size_x - 1))

            # Map Pixel Y to a natural number [0, grid_size_y - 1] (Inverting so 0 is at bottom)
            ry_float = (1.0 - (self.r0_px.y() / self.canvas_height)) * self.grid_size_y
            ry = int(np.clip(ry_float, 0, self.grid_size_y - 1))

            r0 = np.array([rx, ry])

            # k0 continues to represent a continuous vector
            kx = (self.k0_tip_px.x() - self.r0_px.x()) * 0.1
            ky = -(self.k0_tip_px.y() - self.r0_px.y()) * 0.1
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

        # Emit all parameters to the main window
        self.setup_saved.emit(potential, r0, k0, sigma_matrix, mass)
        self.accept()
