# ==============================================================================
# ### --- FILE frontend/settings.py --- ###
# ==============================================================================

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)


class Settings(QDialog):
    """
    Dialog window for adjusting purely visual rendering settings.
    Changing these will not trigger a simulation recalculation.
    """

    # --- Class Fields ---
    # Emits: (fps, z_scale, z_offset, fine_scale, z_pot_scale, brightness,
    #        potential_alpha, zoom_order)
    settings_saved = pyqtSignal(int, float, float, int, float, float, float, int)

    fps_spin: QSpinBox
    fine_scale_spin: QSpinBox
    zoom_order_spin: QSpinBox
    z_scale_spin: QDoubleSpinBox
    z_pot_scale_spin: QDoubleSpinBox
    z_offset_spin: QDoubleSpinBox
    brightness_spin: QDoubleSpinBox
    alpha_spin: QDoubleSpinBox

    def __init__(
        self,
        current_fps: int,
        current_z_scale: float,
        current_z_offset: float,
        current_fine_scale: int,
        current_z_pot_scale: float,
        current_brightness: float,
        current_potential_alpha: float,
        current_zoom_order: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the configuration dialog with current default values.

        Args:
            current_fps (int): Current playback FPS used by the animation timer.
            current_z_scale (float): Current vertical amplitude multiplier for the wave.
            current_z_offset (float): Current vertical offset for the potential landscape.
            current_fine_scale (int): Current interpolation multiplier for visual smoothness.
            current_z_pot_scale (float): Current vertical amplitude multiplier for the potential mesh.
            current_brightness (float): Current exposure multiplier for the wave packet colors.
            current_potential_alpha (float): Current transparency level for the potential mesh (0.0 to 1.0).
            current_zoom_order (int): Current spline order (1-5) used when upscaling the wave to the fine mesh.
            parent (Optional[QWidget]): Unused; transient parent is set by MainWindow
                after the window is shown.
        """
        super().__init__(None)

        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        self.setWindowTitle("Visual Settings")

        layout = QFormLayout(self)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(current_fps)
        layout.addRow("Playback FPS:", self.fps_spin)

        # Set maximum "coarse to fine" grid scale for interpolation
        self.fine_scale_spin = QSpinBox()
        self.fine_scale_spin.setRange(1, 10)
        self.fine_scale_spin.setValue(current_fine_scale)
        layout.addRow("Interpolation Scale (Coarse -> Fine):", self.fine_scale_spin)

        # B-spline order for the coarse -> fine upscale. 1 is fastest but
        # faceted, 2 is the recommended sweet spot, 3+ is smoother but slower
        # and may overshoot near sharp probability peaks.
        self.zoom_order_spin = QSpinBox()
        self.zoom_order_spin.setRange(1, 5)
        self.zoom_order_spin.setValue(current_zoom_order)
        self.zoom_order_spin.setToolTip(
            "B-spline order for the wave upscale.\n"
            "1 = linear (fastest, faceted)\n"
            "2 = quadratic (recommended)\n"
            "3+ = cubic+ (smoother, slower, can overshoot)"
        )
        layout.addRow("Interpolation Order:", self.zoom_order_spin)

        # Set maximum z scale for the probability wave
        self.z_scale_spin = QDoubleSpinBox()
        self.z_scale_spin.setRange(1, 1000.0)
        self.z_scale_spin.setValue(current_z_scale)
        self.z_scale_spin.setSingleStep(1.0)
        layout.addRow("Wave Amplitude (Z Scale):", self.z_scale_spin)

        # Set maximum z scale for the potential energy barrier
        self.z_pot_scale_spin = QDoubleSpinBox()
        self.z_pot_scale_spin.setRange(0.01, 50.0)
        self.z_pot_scale_spin.setValue(current_z_pot_scale)
        self.z_pot_scale_spin.setSingleStep(0.01)
        layout.addRow("Potential Amplitude (Z Scale):", self.z_pot_scale_spin)

        # Set maximum vertical offset to physically move the potential down/up
        self.z_offset_spin = QDoubleSpinBox()
        self.z_offset_spin.setRange(-50.0, 50.0)
        self.z_offset_spin.setValue(current_z_offset)
        self.z_offset_spin.setSingleStep(0.5)
        layout.addRow("Potential Depth Offset:", self.z_offset_spin)

        # Multiplier to increase the brightness/visibility of the probability tails
        self.brightness_spin = QDoubleSpinBox()
        # Strictly enforce minimum value so that the wave function doesn't disappear
        self.brightness_spin.setMinimum(0.1)
        self.brightness_spin.setRange(0.1, 1000.0)
        self.brightness_spin.setValue(current_brightness)
        self.brightness_spin.setSingleStep(5.0)
        layout.addRow("Brightness Multiplier:", self.brightness_spin)

        # Transparency level for the drawn potential fields
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setValue(current_potential_alpha)
        self.alpha_spin.setSingleStep(0.05)
        layout.addRow("Potential Alpha (Transparency):", self.alpha_spin)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        update_btn = btn_box.addButton("Update", QDialogButtonBox.ButtonRole.ApplyRole)
        if update_btn is not None:
            update_btn.clicked.connect(self.apply_settings)
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _emit_current_values(self) -> None:
        """Push the current spin-box values to the main window."""
        self.settings_saved.emit(
            self.fps_spin.value(),
            self.z_scale_spin.value(),
            self.z_offset_spin.value(),
            self.fine_scale_spin.value(),
            self.z_pot_scale_spin.value(),
            self.brightness_spin.value(),
            self.alpha_spin.value(),
            self.zoom_order_spin.value(),
        )

    def apply_settings(self) -> None:
        """
        Applies the current values to the running animation without closing
        this dialog (e.g. preview interpolation changes immediately).
        """
        self._emit_current_values()

    def save_settings(self) -> None:
        """Applies settings and closes the dialog."""
        self._emit_current_values()
        self.accept()
