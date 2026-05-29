# ==============================================================================
# ### --- FILE frontend/animation_controls_widget.py --- ###
# ==============================================================================

from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
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
    reset_camera_requested = pyqtSignal()
    open_setup_requested = pyqtSignal()
    open_settings_requested = pyqtSignal()
    toggle_potential_requested = pyqtSignal(bool)
    stop_calculation_requested = pyqtSignal()

    total_frames: int
    fps: int
    timer: QTimer
    potential_visible: bool
    play_pause_btn: QPushButton
    time_label: QLabel
    slider: QSlider
    toggle_pot_btn: QPushButton
    setup_btn: QPushButton
    settings_btn: QPushButton
    stop_calc_btn: QPushButton

    def __init__(
        self,
        total_frames: int,
        fps: int,
        time_per_frame: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the playback controls widget.

        Args:
            total_frames (int): Total number of frames in the animation buffer.
            fps (int): Frames per second playback rate.
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.total_frames = total_frames
        self.fps = fps
        self.time_per_frame = time_per_frame

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

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_pause_btn)

        self.time_label = QLabel("Time: 0.000 a. u.  Frame: 0")
        layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        self.reset_cam_btn = QPushButton("Reset Camera")
        self.reset_cam_btn.clicked.connect(self.reset_camera_requested.emit)
        layout.addWidget(self.reset_cam_btn)

        self.toggle_pot_btn = QPushButton("Hide Potential")
        self.toggle_pot_btn.clicked.connect(self.toggle_potential)
        layout.addWidget(self.toggle_pot_btn)

        self.setup_btn = QPushButton("Simulation Setup")
        self.setup_btn.clicked.connect(self.open_setup_requested.emit)
        layout.addWidget(self.setup_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_requested.emit)
        layout.addWidget(self.settings_btn)

        self.stop_calc_btn = QPushButton("Stop Calculation")
        self.stop_calc_btn.clicked.connect(self.stop_calculation_requested.emit)
        self.stop_calc_btn.setVisible(False)
        layout.addWidget(self.stop_calc_btn)

        # Keyboard shortcuts for play/pause and fast-forwarding
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_space.activated.connect(self.play_pause_btn.click)

        SKIP_FRAMES_AMOUNT = 30
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_right.activated.connect(
            lambda: self.move_to_frame(self.slider.value() + SKIP_FRAMES_AMOUNT)
        )

        self.shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_left.activated.connect(
            lambda: self.move_to_frame(self.slider.value() - SKIP_FRAMES_AMOUNT)
        )

        # Ensure highest priority for the shortcuts to avoid conflicts with other widgets
        self.shortcut_space.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_right.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_left.setContext(Qt.ShortcutContext.WindowShortcut)

    def enter_calculating_mode(self) -> None:
        """Disable playback controls and show the stop-calculation button."""
        self.pause()
        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)
        self.toggle_pot_btn.setEnabled(False)
        self.setup_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.stop_calc_btn.setVisible(True)
        self.stop_calc_btn.setEnabled(True)

    def exit_calculating_mode(self) -> None:
        """Restore normal playback controls after calculation finishes or is stopped."""
        self.stop_calc_btn.setVisible(False)
        self.play_pause_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.toggle_pot_btn.setEnabled(True)
        self.setup_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)

    def toggle_play_pause(self) -> None:
        """Starts or pauses animation depending on current playback state."""
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        """Starts animation playback. Resets to frame 0 if at the final frame."""
        if self.slider.value() >= self.total_frames - 1:
            self.slider.setValue(0)
        # Wait for the delay duration (in milliseconds) before emitting the second frame
        self.timer.start(1000 // self.fps)
        self.play_pause_btn.setText("Pause")

    def pause(self) -> None:
        """Pauses animation playback by stopping the internal timer."""
        self.timer.stop()
        self.play_pause_btn.setText("Play")

    def advance_frame(self) -> None:
        """Advances playback to the next frame or pauses if the simulation ends."""
        current_frame: int = self.slider.value()
        if current_frame < self.total_frames - 1:
            self.slider.setValue(current_frame + 1)
        else:
            self.pause()

    def move_to_frame(self, frame: int) -> None:
        """Moves the playback slider to a specific frame index."""
        self.slider.setValue(min(max(frame, 0), self.total_frames - 1))

    def update_time_label(self) -> None:
        """Updates the text of the time label based on current frame and physical time step."""
        current_frame = self.slider.value()
        physical_time = current_frame * self.time_per_frame
        self.time_label.setText(f"Time: {physical_time:.3f} a. u.  Frame: {current_frame}")

    def on_slider_changed(self, value: int) -> None:
        """Handles slider position changes, updates UI label text, and emits current frame index."""
        self.update_time_label()  # <-- Używa nowej funkcji
        self.frame_changed.emit(value)

    def update_settings(
        self, fps: int, total_frames: int, time_per_frame: Optional[float] = None
    ) -> None:
        """Updates internal playback settings and smoothly adjusts the active timer interval."""
        self.fps = fps
        self.total_frames = total_frames
        if time_per_frame is not None:
            self.time_per_frame = time_per_frame  # <-- DODANE

        self.slider.setRange(0, self.total_frames - 1)
        self.update_time_label()  # <-- Odświeżenie etykiety po zmianie ustawień

        if self.timer.isActive():
            self.timer.setInterval(1000 // self.fps)

    def toggle_potential(self) -> None:
        """Toggles the visibility state of the potential and updates the button text."""
        self.potential_visible = not self.potential_visible

        if self.potential_visible:
            self.toggle_pot_btn.setText("Hide Potential")
        else:
            self.toggle_pot_btn.setText("Show Potential")

        self.toggle_potential_requested.emit(self.potential_visible)
