# ==============================================================================
# ### --- FILE frontend/animation_controls_widget.py --- ###
# ==============================================================================

from typing import Optional

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

    # --- Class Fields ---
    frame_changed = pyqtSignal(int)
    open_setup_requested = pyqtSignal()
    open_settings_requested = pyqtSignal()
    toggle_potential_requested = pyqtSignal(bool)

    total_frames: int
    fps: int
    timer: QTimer
    potential_visible: bool
    play_btn: QPushButton
    pause_btn: QPushButton
    time_label: QLabel
    slider: QSlider
    toggle_pot_btn: QPushButton
    setup_btn: QPushButton
    settings_btn: QPushButton

    def __init__(
        self, total_frames: int, fps: int, parent: Optional[QWidget] = None
    ) -> None:
        """
        Initializes the playback controls widget.

        Args:
            total_frames (int): Total number of frames in the animation buffer.
            fps (int): Frames per second playback rate.
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.total_frames: int = total_frames
        self.fps: int = fps

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.potential_visible: bool = (
            True  # Track the current state of the potential mesh
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Sets up buttons, labels, and the horizontal slider layout."""
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

        self.toggle_pot_btn = QPushButton("Hide Potential")
        self.toggle_pot_btn.clicked.connect(self.toggle_potential)
        layout.addWidget(self.toggle_pot_btn)

        self.setup_btn = QPushButton("Simulation Setup")
        self.setup_btn.clicked.connect(self.open_setup_requested.emit)
        layout.addWidget(self.setup_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_requested.emit)
        layout.addWidget(self.settings_btn)

    def play(self) -> None:
        """Starts animation playback. Resets to frame 0 if at the final frame."""
        if self.slider.value() >= self.total_frames - 1:
            self.slider.setValue(0)
        self.timer.start(1000 // self.fps)

    def pause(self) -> None:
        """Pauses animation playback by stopping the internal timer."""
        self.timer.stop()

    def advance_frame(self) -> None:
        """Advances playback to the next frame or pauses if the simulation ends."""
        current_frame: int = self.slider.value()
        if current_frame < self.total_frames - 1:
            self.slider.setValue(current_frame + 1)
        else:
            self.pause()

    def on_slider_changed(self, value: int) -> None:
        """Handles slider position changes, updates UI label text, and emits current frame index."""
        self.time_label.setText(f"Time: {value}")
        self.frame_changed.emit(value)

    def toggle_potential(self) -> None:
        """Toggles the visibility state of the potential and updates the button text."""
        self.potential_visible = not self.potential_visible

        if self.potential_visible:
            self.toggle_pot_btn.setText("Hide Potential")
        else:
            self.toggle_pot_btn.setText("Show Potential")

        self.toggle_potential_requested.emit(self.potential_visible)

    def update_settings(self, fps: int, total_frames: int) -> None:
        """Updates internal playback settings and smoothly adjusts the active timer interval."""
        self.fps = fps
        self.total_frames = total_frames
        self.slider.setRange(0, self.total_frames - 1)

        if self.timer.isActive():
            self.timer.setInterval(1000 // self.fps)
