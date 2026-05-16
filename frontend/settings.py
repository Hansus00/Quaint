# ==============================================================================
# ### --- FILE frontend/settings.py --- ###
# ==============================================================================

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
)


class Settings(QDialog):
    """
    Dialog window for adjusting playback, resolution, and visualization settings.
    """

    # Emits: (fps, total_frames, size_x, size_y, z_scale, z_offset, fine_scale, z_pot_scale)
    settings_saved = pyqtSignal(int, int, int, int, float, float, int, float)

    def __init__(
        self,
        current_fps: int,
        current_total_frames: int,
        current_size_x: int,
        current_size_y: int,
        current_z_scale: float,
        current_z_offset: float,
        current_fine_scale: int,
        current_z_pot_scale: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the configuration dialog with current default values.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")

        layout = QFormLayout(self)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(current_fps)
        layout.addRow("Frames Per Second (FPS):", self.fps_spin)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 10000)
        self.frames_spin.setValue(current_total_frames)
        layout.addRow("Total Frames:", self.frames_spin)

        self.size_x_spin = QSpinBox()
        self.size_x_spin.setRange(10, 500)
        self.size_x_spin.setValue(current_size_x)
        layout.addRow("Grid Resolution X:", self.size_x_spin)

        self.size_y_spin = QSpinBox()
        self.size_y_spin.setRange(10, 500)
        self.size_y_spin.setValue(current_size_y)
        layout.addRow("Grid Resolution Y:", self.size_y_spin)

        self.fine_scale_spin = QSpinBox()
        self.fine_scale_spin.setRange(1, 10)
        self.fine_scale_spin.setValue(current_fine_scale)
        layout.addRow("Interpolation Scale (Coarse -> Fine):", self.fine_scale_spin)

        self.z_scale_spin = QDoubleSpinBox()
        self.z_scale_spin.setRange(0.1, 100.0)
        self.z_scale_spin.setValue(current_z_scale)
        self.z_scale_spin.setSingleStep(1.0)
        layout.addRow("Wave Amplitude (Z Scale):", self.z_scale_spin)

        self.z_pot_scale_spin = QDoubleSpinBox()
        self.z_pot_scale_spin.setRange(0.01, 50.0)
        self.z_pot_scale_spin.setValue(current_z_pot_scale)
        self.z_pot_scale_spin.setSingleStep(0.01)
        layout.addRow("Potential Amplitude (Z Scale):", self.z_pot_scale_spin)

        self.z_offset_spin = QDoubleSpinBox()
        self.z_offset_spin.setRange(-50.0, 50.0)
        self.z_offset_spin.setValue(current_z_offset)
        self.z_offset_spin.setSingleStep(0.5)
        layout.addRow("Potential Depth Offset:", self.z_offset_spin)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def save_settings(self) -> None:
        """Emits the updated setting values and closes the dialog indicating acceptance."""
        self.settings_saved.emit(
            self.fps_spin.value(),
            self.frames_spin.value(),
            self.size_x_spin.value(),
            self.size_y_spin.value(),
            self.z_scale_spin.value(),
            self.z_offset_spin.value(),
            self.fine_scale_spin.value(),
            self.z_pot_scale_spin.value(),
        )
        self.accept()
