# ==============================================================================
# ### --- FILE setup_drawer.py --- ###
# ==============================================================================

from typing import Optional
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QComboBox,
    QWidget
)


class SetupDrawer(QDialog):
    """
    A 2D canvas dialog for drawing a potential field and setting wavepacket starting states.
    Emits a tuple upon saving containing: (potential_matrix, r0, k0, sigma_matrix, mass)
    """

    simulation_changed = pyqtSignal(str)
    setup_saved = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)

    def __init__(
        self, 
        grid_size_x: int = 25, 
        grid_size_y: int = 35, 
        x_limit: float = 5.0, 
        y_limit: float = 5.0, 
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initializes the drawing canvas.

        Args:
            grid_size_x (int): Horizontal resolution of the simulation grid.
            grid_size_y (int): Vertical resolution of the simulation grid.
            x_limit (float): Maximum physical coordinate in X.
            y_limit (float): Maximum physical coordinate in Y.
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Setup: Potential & Wavepacket")

        self.grid_size_x: int = grid_size_x
        self.grid_size_y: int = grid_size_y
        self.x_limit: float = x_limit
        self.y_limit: float = y_limit

        self.canvas_width: int = 400
        self.canvas_height: int = int(400 * (grid_size_y / grid_size_x))

        self.image = QImage(
            self.canvas_width, self.canvas_height, QImage.Format.Format_ARGB32
        )
        self.image.fill(Qt.GlobalColor.white)

        # State Variables
        self.drawing_potential: bool = False
        self.last_point: QPoint = QPoint()
        self.mode: str = "potential"  # Options: "potential", "wavepacket"

        # Wavepacket vector state (stored in pixel coordinates for the UI)
        self.r0_px: Optional[QPoint] = None
        self.k0_tip_px: Optional[QPoint] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Sets up the radio buttons, input fields, and layouts for the canvas dialog."""
        layout = QVBoxLayout(self)

        # Mode Selection
        mode_layout = QHBoxLayout()
        self.radio_pot = QRadioButton("Draw Potential")
        self.radio_pot.setChecked(True)
        self.radio_wave = QRadioButton("Set Wavepacket (r0, k0)")

        self.radio_pot.toggled.connect(self.update_mode)

        self.simulation_menu = QComboBox()
        self.simulation_menu.addItem("Constant")
        self.simulation_menu.addItem("Crank-Nicolson")
        self.simulation_menu.addItem("SSFM")
        self.simulation_menu.currentTextChanged.connect(self.simulation_changed.emit)
        mode_layout.addWidget(QLabel("Simulation Method:"))
        mode_layout.addWidget(self.simulation_menu)

        mode_layout.addWidget(self.radio_pot)
        mode_layout.addWidget(self.radio_wave)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Physics Parameters (Sigma Matrix & Mass)
        params_layout = QHBoxLayout()

        # Sigma xx
        params_layout.addWidget(QLabel("s_xx:"))
        self.sig_xx_input = QDoubleSpinBox()
        self.sig_xx_input.setRange(0.1, 20.0)
        self.sig_xx_input.setValue(1.0)
        self.sig_xx_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xx_input)

        # Sigma xy
        params_layout.addWidget(QLabel("s_xy:"))
        self.sig_xy_input = QDoubleSpinBox()
        self.sig_xy_input.setRange(-10.0, 10.0)
        self.sig_xy_input.setValue(0.0)
        self.sig_xy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_xy_input)

        # Sigma yy
        params_layout.addWidget(QLabel("s_yy:"))
        self.sig_yy_input = QDoubleSpinBox()
        self.sig_yy_input.setRange(0.1, 20.0)
        self.sig_yy_input.setValue(1.0)
        self.sig_yy_input.setSingleStep(0.1)
        params_layout.addWidget(self.sig_yy_input)

        # Mass
        params_layout.addWidget(QLabel("m:"))
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.01, 100.0)
        self.mass_input.setValue(1.0)
        self.mass_input.setSingleStep(0.1)
        params_layout.addWidget(self.mass_input)

        layout.addLayout(params_layout)

        # Canvas constraints (increased height slightly for new UI row)
        self.setFixedSize(self.canvas_width, self.canvas_height + 130)

        # Action Controls
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Potential")
        clear_btn.clicked.connect(self.clear_canvas)
        save_btn = QPushButton("Save & Update Simulation")
        save_btn.clicked.connect(self.save_and_close)

        controls.addWidget(clear_btn)
        controls.addWidget(save_btn)
        layout.addStretch()
        layout.addLayout(controls)

    def update_mode(self) -> None:
        """Updates the drawing state based on radio button selection."""
        if self.radio_pot.isChecked():
            self.mode = "potential"
        else:
            self.mode = "wavepacket"

    def paintEvent(self, event) -> None:
        """Handles the rendering of the canvas and overlay objects like the wave vector."""
        canvas_painter = QPainter(self)

        # Offset Y to account for top UI controls (now roughly 80px)
        offset_y = 80
        canvas_painter.translate(0, offset_y)
        canvas_painter.drawImage(0, 0, self.image)

        if self.r0_px and self.k0_tip_px:
            pen = QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine)
            canvas_painter.setPen(pen)
            canvas_painter.setBrush(Qt.GlobalColor.red)
            canvas_painter.drawLine(self.r0_px, self.k0_tip_px)
            canvas_painter.drawEllipse(self.r0_px, 4, 4)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint() - QPoint(0, 80)

        if pos.y() < 0 or pos.y() > self.canvas_height:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "potential":
                self.drawing_potential = True
                self.last_point = pos
            elif self.mode == "wavepacket":
                self.r0_px = pos
                self.k0_tip_px = pos
                self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint() - QPoint(0, 80)

        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.mode == "potential" and self.drawing_potential:
                painter = QPainter(self.image)
                pen = QPen(
                    QColor(0, 0, 0, 15),
                    30,
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
        """Clears the drawn potential to a blank white canvas."""
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def save_and_close(self) -> None:
        """
        Parses canvas drawing and physics inputs, emits them, and closes the dialog.
        """
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
        ptr = gray_img.constBits()
        ptr.setsize(height * bpl)

        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bpl))
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