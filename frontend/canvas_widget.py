# ==============================================================================
# ### --- FILE frontend/canvas_widget.py --- ###
# ==============================================================================

from typing import Optional
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QResizeEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class AspectRatioContainer(QWidget):
    """
    A layout wrapper widget that forces its central child to maintain a strict 
    aspect ratio dynamically, responding organically to parent window resizes.
    """
    def __init__(self, widget: QWidget, aspect_ratio: float, parent: Optional[QWidget] = None) -> None:
        """
        Initializes the dynamic framing container.

        Args:
            widget (QWidget): The inner child widget to constrain.
            aspect_ratio (float): The target Height/Width ratio to lock (e.g. grid_y / grid_x).
        """
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.child_widget = widget

        # Wrap the child in a layout with 0 initial margins
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.child_widget)

    def set_aspect_ratio(self, aspect_ratio: float) -> None:
        """Updates the locked ratio and triggers an immediate geometry recalculation."""
        self.aspect_ratio = aspect_ratio
        from PyQt6.QtGui import QResizeEvent
        self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Calculates the largest bounded box fitting the ratio and pads the excess 
        space with dynamic layout margins to strictly center the child canvas.
        """
        w = event.size().width()
        h = event.size().height()

        if w == 0 or h == 0:
            return super().resizeEvent(event)

        if h / w > self.aspect_ratio:
            # Layout is too tall, limit by width
            new_w = w
            new_h = int(w * self.aspect_ratio)
        else:
            # Layout is too wide, limit by height
            new_h = h
            new_w = int(h / self.aspect_ratio)

        margin_x = (w - new_w) // 2
        margin_y = (h - new_h) // 2

        self._layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        super().resizeEvent(event)


class CanvasWidget(QWidget):
    """
    An isolated interactive 2D canvas element dedicated entirely to graphical user input.
    It operates in its own localized pixel coordinate space and adapts seamlessly to layout resizes.
    """

    def __init__(self, width: int, height: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Allow the widget to shrink and expand freely within the container
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # The underlying image layout storing the raw potential configuration
        self.image = QImage(width, height, QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.white)

        # Current interaction mode state
        self.mode: str = "brush"
        self.drawing_potential: bool = False
        self.last_point: QPoint = QPoint()

        self.r0_px: Optional[QPoint] = None
        self.k0_tip_px: Optional[QPoint] = None

        self.brush_strength: int = 15
        self.brush_width: int = 30

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Automatically triggered by the Qt framework whenever the widget boundaries change.
        Rescales the internal image and adjusts wavepacket vectors proportionally.
        """
        new_size = event.size()
        old_size = event.oldSize()

        if old_size.isValid() and old_size.width() > 0 and old_size.height() > 0:
            self.image = self.image.scaled(
                new_size.width(),
                new_size.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,  # Safe because Container enforces ratio
                Qt.TransformationMode.SmoothTransformation,
            )

            # Move anchors proportionally to remain physically accurate visually
            if self.r0_px:
                self.r0_px.setX(int((self.r0_px.x() / old_size.width()) * new_size.width()))
                self.r0_px.setY(int((self.r0_px.y() / old_size.height()) * new_size.height()))
            if self.k0_tip_px:
                self.k0_tip_px.setX(int((self.k0_tip_px.x() / old_size.width()) * new_size.width()))
                self.k0_tip_px.setY(int((self.k0_tip_px.y() / old_size.height()) * new_size.height()))
        else:
            self.image = self.image.scaled(
                new_size.width(),
                new_size.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        super().resizeEvent(event)

    def set_image(self, img: QImage) -> None:
        """Directly loads an external predefined image pattern onto the workspace view."""
        self.image = img
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

        if self.r0_px and self.k0_tip_px:
            pen = QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.GlobalColor.red)
            painter.drawLine(self.r0_px, self.k0_tip_px)
            painter.drawEllipse(self.r0_px, 4, 4)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser"):
                self.drawing_potential = True
                self.last_point = pos
            elif self.mode == "wavepacket":
                self.r0_px = pos
                self.k0_tip_px = pos
                self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser") and self.drawing_potential:
                painter = QPainter(self.image)
                if self.mode == "brush":
                    color = QColor(0, 0, 0, self.brush_strength)
                else:
                    color = QColor(255, 255, 255, 255)

                pen = QPen(
                    color,
                    self.brush_width,
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