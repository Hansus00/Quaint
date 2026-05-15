# ==============================================================================
# ### --- FILE animation_controls_widget.py --- ###
# ==============================================================================

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class AnimationControlsWidget(QWidget):
    """
    Widget component containing playback UI and timer logic.
    """

    frame_changed = pyqtSignal(int)
    open_setup_requested = pyqtSignal()
    open_settings_requested = pyqtSignal()

    def __init__(self, total_frames, fps, parent=None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.fps = fps

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause)
        layout.addWidget(self.pause_btn)

        self.time_label = QLabel("Time: 0")
        layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        self.setup_btn = QPushButton("Simulation Setup")
        self.setup_btn.clicked.connect(self.open_setup_requested.emit)
        layout.addWidget(self.setup_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_requested.emit)
        layout.addWidget(self.settings_btn)

    def play(self):
        if self.slider.value() >= self.total_frames - 1:
            self.slider.setValue(0)
        self.timer.start(1000 // self.fps)

    def pause(self):
        self.timer.stop()

    def advance_frame(self):
        current_frame = self.slider.value()
        if current_frame < self.total_frames - 1:
            self.slider.setValue(current_frame + 1)
        else:
            self.pause()

    def on_slider_changed(self, value):
        self.time_label.setText(f"Time: {value}")
        self.frame_changed.emit(value)

    def update_settings(self, fps, total_frames):
        self.fps = fps
        self.total_frames = total_frames
        self.slider.setRange(0, self.total_frames - 1)

        # If currently playing, update the timer interval smoothly
        if self.timer.isActive():
            self.timer.setInterval(1000 // self.fps)
