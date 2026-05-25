# ==============================================================================
# ### --- FILE frontend/canvas_widget.py --- ###
# ==============================================================================

from typing import Optional
import numpy as np

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


class AspectRatioContainer(QWidget):
    """
    A layout wrapper widget that forces its central child to maintain a strict
    aspect ratio dynamically, responding organically to parent window resizes.
    """

    # --- Class Fields ---
    aspect_ratio: float
    child_widget: QWidget
    _layout: QVBoxLayout

    def __init__(
        self, widget: QWidget, aspect_ratio: float, parent: Optional[QWidget] = None
    ) -> None:
        """
        Initializes the dynamic framing container.

        Args:
            widget (QWidget): The inner child widget to constrain.
            aspect_ratio (float): The target Height/Width ratio to lock (e.g. grid_y / grid_x).
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.child_widget = widget

        # Wrap the child in a layout with 0 initial margins
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.child_widget)

    def set_aspect_ratio(self, aspect_ratio: float) -> None:
        """
        Updates the locked ratio and triggers an immediate geometry recalculation.

        Args:
            aspect_ratio (float): The new Height/Width ratio to enforce.
        """
        self.aspect_ratio = aspect_ratio
        from PyQt6.QtGui import QResizeEvent

        self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """
        Calculates the largest bounded box fitting the ratio and pads the excess
        space with dynamic layout margins to strictly center the child canvas.

        Args:
            a0 (QResizeEvent | None): The resize event containing new and old dimensions.
        """
        if a0 is None:
            return super().resizeEvent(a0)

        w = a0.size().width()
        h = a0.size().height()

        if w == 0 or h == 0:
            return super().resizeEvent(a0)

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
        super().resizeEvent(a0)


class CanvasWidget(QWidget):
    """
    Interactive 2D canvas backing the potential drawer. The underlying QImage is
    stored at the native physics-grid resolution (grid_size_x by grid_size_y);
    on screen the image is merely upscaled for display. Wavepacket anchors are
    likewise kept in grid coordinates (Qt's top-to-bottom Y), so opening and
    saving the dialog without changes is a lossless identity operation.
    """

    # --- Class Fields ---
    grid_size_x: int
    grid_size_y: int
    image: QImage
    mode: str
    drawing_potential: bool
    last_point_grid: QPointF
    r0_grid: Optional[QPointF]
    k0_tip_grid: Optional[QPointF]
    brush_strength: int
    brush_width: int

    def __init__(
        self,
        grid_size_x: int,
        grid_size_y: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the drawing surface at the simulation grid resolution.

        Args:
            grid_size_x (int): Horizontal physics grid resolution (canvas width in pixels).
            grid_size_y (int): Vertical physics grid resolution (canvas height in pixels).
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Allow the widget to shrink and expand freely within the container
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # The underlying image lives in grid pixel space; one pixel per simulation cell.
        self.image = QImage(grid_size_x, grid_size_y, QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.white)

        self.mode = "brush"
        self.drawing_potential = False
        self.last_point_grid = QPointF()

        self.r0_grid = None
        self.k0_tip_grid = None

        self.brush_strength = 15
        # Brush diameter measured in grid cells (matches underlying image scale).
        self.brush_width = 3

        # Enable mouse tracking to paint the brush preview
        self.setMouseTracking(True)
        self.current_hover_grid = None

    def _widget_to_grid(self, p: QPoint) -> QPointF:
        """Convert a widget pixel coordinate to fractional grid coordinates."""
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        return QPointF(
            p.x() * self.grid_size_x / w,
            p.y() * self.grid_size_y / h,
        )

    def _grid_to_widget(self, p: QPointF) -> QPointF:
        """Convert a fractional grid coordinate to widget pixel space for rendering."""
        return QPointF(
            p.x() * self.width() / self.grid_size_x,
            p.y() * self.height() / self.grid_size_y,
        )

    def set_image(self, img: QImage) -> None:
        """
        Replace the canvas image. Inputs at the wrong resolution are scaled once
        to match the current grid size; otherwise the image is stored verbatim
        so that open/save round-trips do not interpolate.

        Args:
            img (QImage): New potential image to display.
        """
        if img.width() != self.grid_size_x or img.height() != self.grid_size_y:
            img = img.scaled(
                self.grid_size_x,
                self.grid_size_y,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.image = img.convertToFormat(QImage.Format.Format_ARGB32)
        self.update()

    def set_grid_size(self, grid_size_x: int, grid_size_y: int) -> None:
        """
        Rebind the canvas to a new grid resolution. Both the stored image and
        the wavepacket anchors are rescaled proportionally so that visible
        content stays in place across a resolution change.

        Args:
            grid_size_x (int): New horizontal grid resolution.
            grid_size_y (int): New vertical grid resolution.
        """
        if grid_size_x == self.grid_size_x and grid_size_y == self.grid_size_y:
            return

        new_img = self.image.scaled(
            grid_size_x,
            grid_size_y,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ).convertToFormat(QImage.Format.Format_ARGB32)

        scale_x = grid_size_x / self.grid_size_x
        scale_y = grid_size_y / self.grid_size_y
        if self.r0_grid is not None:
            self.r0_grid = QPointF(
                self.r0_grid.x() * scale_x,
                self.r0_grid.y() * scale_y,
            )
        if self.k0_tip_grid is not None:
            self.k0_tip_grid = QPointF(
                self.k0_tip_grid.x() * scale_x,
                self.k0_tip_grid.y() * scale_y,
            )

        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.image = new_img
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """
        Renders the active canvas element, superimposing interactive vector overlays.
        The image is upscaled to the current widget size for display only.

        Args:
            a0: The QPaintEvent triggered by the Qt framework.
        """
        painter = QPainter(self)
        scaled = self.image.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, scaled)

        # Brush / eraser preview rendering
        if self.mode in ("brush", "eraser") and self.current_hover_grid is not None:
            # Convert the current hover position to widget coordinates for rendering the preview
            hover_w = self._grid_to_widget(self.current_hover_grid)
            
            scale_ratio = self.width() / self.grid_size_x
            preview_radius = (self.brush_width * scale_ratio) / 2.0

            if self.mode == "brush":
                # Preview alpha dependent on brush strength
                preview_color = QColor(0, 0, 0,  np.clip(self.brush_strength * 3, 0, 255))
                painter.setBrush(preview_color)
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                # Drawing preview for eraser as well
                painter.setBrush(QColor(255, 255, 255, 150))
                painter.setPen(QPen(Qt.GlobalColor.black, 1))

            painter.drawEllipse(hover_w, preview_radius, preview_radius)

        if self.r0_grid is not None and self.k0_tip_grid is not None:
            r0_w = self._grid_to_widget(self.r0_grid)
            k0_w = self._grid_to_widget(self.k0_tip_grid)
            pen = QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.GlobalColor.red)
            painter.drawLine(r0_w, k0_w)
            painter.drawEllipse(r0_w, 4, 4)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """
        Image resolution is locked to the physics grid, so no rescale of the
        underlying QImage is performed on widget resize - only the on-screen
        rendering rescales (cf. ``paintEvent``).

        Args:
            a0 (QResizeEvent | None): The resize event containing new and old dimensions.
        """
        super().resizeEvent(a0)

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        """
        Captures user interaction starts, anchoring the drawing tool or setting state coordinates.

        Args:
            a0: The QMouseEvent containing the click position and button state.
        """
        if a0 is None:
            return super().mousePressEvent(a0)

        pos_grid = self._widget_to_grid(a0.position().toPoint())
        if a0.button() == Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser"):
                self.drawing_potential = True
                self.last_point_grid = pos_grid
            elif self.mode == "wavepacket":
                self.r0_grid = pos_grid
                self.k0_tip_grid = pos_grid
                self.update()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        """
        Processes drag inputs, drawing stroke pathways or modifying the active state vector length.

        Args:
            a0: The QMouseEvent containing the cursor position.
        """
        if a0 is None:
            return super().mouseMoveEvent(a0)

        pos_grid = self._widget_to_grid(a0.position().toPoint())
        self.current_hover_grid = pos_grid

        if a0.buttons() & Qt.MouseButton.LeftButton:
            if self.mode in ("brush", "eraser") and self.drawing_potential:
                painter = QPainter(self.image)
                if self.mode == "brush":
                    color = QColor(0, 0, 0, self.brush_strength)
                else:
                    color = QColor(255, 255, 255, 255)

                pen = QPen(
                    color,
                    max(1.0, float(self.brush_width)),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(pen)
                painter.drawLine(self.last_point_grid, pos_grid)
                self.last_point_grid = pos_grid

            elif self.mode == "wavepacket":
                self.k0_tip_grid = pos_grid
            
        self.update()
        
    def leaveEvent(self, a0) -> None:
        """
        Clears the hover preview when the mouse leaves the canvas bounds.

        Args:
            a0: The QEvent containing the cursor position.
        """
        self.current_hover_grid = None
        self.update()
        super().leaveEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        """
        Closes active input streams upon releasing mouse actions.

        Args:
            a0: The QMouseEvent triggering the release.
        """
        if a0 is None:
            return super().mouseReleaseEvent(a0)

        if a0.button() == Qt.MouseButton.LeftButton:
            self.drawing_potential = False
