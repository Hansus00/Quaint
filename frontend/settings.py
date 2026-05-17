# ==============================================================================
# ### --- FILE frontend/settings.py --- ###
# ==============================================================================

from typing import Optional

from PyQt6.QtCore import pyqtSignal
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
    # Emits: (z_scale, z_offset, fine_scale, z_pot_scale, brightness)
    settings_saved = pyqtSignal(float, float, int, float, float)

    fine_scale_spin: QSpinBox
    z_scale_spin: QDoubleSpinBox
    z_pot_scale_spin: QDoubleSpinBox
    z_offset_spin: QDoubleSpinBox
    brightness_spin: QDoubleSpinBox

    def __init__(
        self,
        current_z_scale: float,
        current_z_offset: float,
        current_fine_scale: int,
        current_z_pot_scale: float,
        current_brightness: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the configuration dialog with current default values.

        Args:
            current_z_scale (float): Current vertical amplitude multiplier for the wave.
            current_z_offset (float): Current vertical offset for the potential landscape.
            current_fine_scale (int): Current interpolation multiplier for visual smoothness.
            current_z_pot_scale (float): Current vertical amplitude multiplier for the potential mesh.
            current_brightness (float): Current exposure multiplier for the wave packet colors.
            parent (Optional[QWidget]): Parent widget to center the dialog on.
        """
        super().__init__(parent)
        self.setWindowTitle("Visual Settings")

        layout = QFormLayout(self)

        # Set maximum "coarse to fine" grid scale for interpolation
        self.fine_scale_spin = QSpinBox()
        self.fine_scale_spin.setRange(1, 10)
        self.fine_scale_spin.setValue(current_fine_scale)
        layout.addRow("Interpolation Scale (Coarse -> Fine):", self.fine_scale_spin)

        # Set maximum z scale for the probability wave
        self.z_scale_spin = QDoubleSpinBox()
        self.z_scale_spin.setRange(0.1, 100.0)
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

        # Standard Ok / Cancel buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def save_settings(self) -> None:
        """
        Emits the updated setting values via the `settings_saved` signal
        and closes the dialog indicating acceptance.
        """
        self.settings_saved.emit(
            self.z_scale_spin.value(),
            self.z_offset_spin.value(),
            self.fine_scale_spin.value(),
            self.z_pot_scale_spin.value(),
            self.brightness_spin.value(),
        )
        self.accept()
