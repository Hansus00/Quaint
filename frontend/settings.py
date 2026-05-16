# ==============================================================================
# ### --- FILE settings.py --- ###
# ==============================================================================

from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)


class Settings(QDialog):
    """
    Dialog window for adjusting playback frames per second (FPS) and total simulation frames.
    """

    settings_saved = pyqtSignal(int, int)  # Emits (fps, total_frames)

    def __init__(self, current_fps: int, current_total_frames: int, parent: Optional[QWidget] = None) -> None:
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

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def save_settings(self) -> None:
        """Emits the updated setting values and closes the dialog indicating acceptance."""
        self.settings_saved.emit(self.fps_spin.value(), self.frames_spin.value())
        self.accept()