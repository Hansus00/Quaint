# ==============================================================================
# ### --- FILE frontend/setup_drawer.py --- ###
# ==============================================================================

from typing import Optional

import logging
import numpy as np
from backend.Potential import (
    EmbeddedPotential,
    GaussianBumpPotential,
    HarmonicPotential,
    Potential,
    WShaped,
)
from backend.StationaryWaveFunc import GaussianPacket
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QFormLayout,
    QScrollArea,
)

from backend.Params import Params, SolverType, WellType
from .canvas_widget import CanvasWidget, AspectRatioContainer
from .simulation_builders import (
    coarse_potential_from_drawer,
    instantiate_solver_with_warnings,
)
from .warning_handler import WarningCaptureHandler

# Conversion factor between an arrow's grid-cell displacement and the physical
# wavevector magnitude. With 1.0 a drag of N grid cells encodes |k| = N, which
# is grid-resolution independent and yields exact round-trips through the
# (canvas <-> Params) boundary.
K_GRID_FACTOR: float = 0.01


class SetupDrawer(QDialog):
    """
    A comprehensive physics configuration dialogue interface linking user sketches to simulation parameters.

    This window encapsulates input form layout parameters (discretization grids, playback rates, particle masses)
    and binds them alongside an interactive canvas element. Upon closure via submission, it performs
    coordinate mapping and grayscale matrix translation to dispatch structured raw simulation data fields
    back to the primary 3D visualization window engine.
    """

    # -- Class Fields --
    # Emits: (method_name)
    simulation_changed = pyqtSignal(str)

    # Emits: (potential_matrix, r0, k0, sigma_matrix, mass, total_frames,
    #         size_x, size_y, delta_t, steps_per_frame, wall_height,
    #         x_limit, y_limit, grid_step, method_name, prebuilt_solver)
    # The prebuilt solver was already constructed for the stability check, so
    # forwarding it lets the main window skip a second expensive build.
    setup_saved = pyqtSignal(
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        int,
        int,
        float,
        int,
        float,
        float,
        float,
        float,
        str,
        object,
    )

    current_frames: int
    grid_size_x: int
    grid_size_y: int
    x_limit: float
    y_limit: float
    initial_sigma: Optional[np.ndarray]
    initial_mass: float
    initial_method: str
    initial_delta_t: float
    initial_steps_per_frame: int
    initial_wall_height: float

    canvas: CanvasWidget
    canvas_container: AspectRatioContainer
    simulation_menu: QComboBox
    preset_menu: QComboBox
    frames_input: QSpinBox
    size_x_input: QSpinBox
    size_y_input: QSpinBox

    delta_t_input: QDoubleSpinBox
    steps_per_frame_input: QSpinBox
    wall_height_input: QDoubleSpinBox

    radio_brush: QRadioButton
    radio_eraser: QRadioButton
    radio_wave: QRadioButton
    sig_xx_input: QDoubleSpinBox
    sig_xy_input: QDoubleSpinBox
    sig_yy_input: QDoubleSpinBox
    mass_input: QDoubleSpinBox
    brush_strength_label: QLabel
    brush_strength_slider: QSlider
    brush_width_label: QLabel
    brush_width_slider: QSlider
    save_btn: QPushButton
    load_params_btn: QPushButton
    save_params_btn: QPushButton

    def __init__(
        self,
        current_frames: int = 150,
        grid_size_x: int = 25,
        grid_size_y: int = 35,
        x_limit: float = 5.0,
        y_limit: float = 5.0,
        initial_grid_step: float = 0.2,
        initial_potential: Optional[np.ndarray] = None,
        initial_r0: Optional[np.ndarray] = None,
        initial_k0: Optional[np.ndarray] = None,
        initial_sigma: Optional[np.ndarray] = None,
        initial_mass: float = 1.0,
        initial_method: str = "Constant",
        initial_delta_t: float = 0.002,
        initial_steps_per_frame: int = 30,
        initial_wall_height: float = 50.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the drawing canvas and internal state.

        Args:
            current_frames (int): Currently active frame buffer duration limit.
            grid_size_x (int): Horizontal resolution of the simulation grid.
            grid_size_y (int): Vertical resolution of the simulation grid.
            x_limit (float): Maximum physical coordinate in X.
            y_limit (float): Maximum physical coordinate in Y.
            initial_potential (Optional[np.ndarray]): Previously saved potential matrix to restore.
            initial_r0 (Optional[np.ndarray]): Previously saved initial position vector.
            initial_k0 (Optional[np.ndarray]): Previously saved initial momentum vector.
            initial_sigma (Optional[np.ndarray]): Previously saved covariance matrix.
            initial_mass (float): Previously saved particle mass.
            initial_method (str): Previously selected simulation method.
            initial_delta_t (float): Previously saved time step size.
            initial_steps_per_frame (int): Previously saved physics sub-steps per animation frame.,
            initial_wall_height (float): Previously saved wall height.
            parent (Optional[QWidget]): Unused; transient parent is set by MainWindow
                after the window is shown.
        """
        super().__init__(None)

        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("Simulation Setup: Potential & Wavepacket")

        # Enforce minimum boundary constraints and add native OS window buttons (Minimize/Maximize)
        self.setMinimumSize(800, 800)
        self.resize(1300, 920)

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        # Capture initial physics environment properties
        self.current_frames = current_frames
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.initial_grid_step = initial_grid_step

        self.initial_sigma = initial_sigma
        self.initial_mass = initial_mass
        self.initial_method = initial_method
        self.initial_delta_t = initial_delta_t
        self.initial_steps_per_frame = initial_steps_per_frame
        self.initial_wall_height = initial_wall_height

        # Canvas image is held at the native physics grid resolution; the widget is
        # purely an upscaled view of those exact (grid_size_x x grid_size_y) pixels.
        self.canvas = CanvasWidget(grid_size_x, grid_size_y)
        self.canvas_container = AspectRatioContainer(
            self.canvas, grid_size_y / grid_size_x
        )

        # Restore past matrix configurations if the user re-enters setup during runtime
        if initial_potential is not None:
            self._restore_canvas(initial_potential)

        self._setup_ui()

        # Restore the wavepacket directly in grid coordinates (no canvas-pixel detour),
        # so opening + saving the dialog untouched is a numerical no-op.
        if initial_r0 is not None and initial_k0 is not None:
            rx_grid = float(initial_r0[0])
            # Physics Y is bottom-up; Qt grid Y is top-down. Flip to draw upright.
            ry_grid = float(self.grid_size_y) - float(initial_r0[1])
            self.canvas.r0_grid = QPointF(rx_grid, ry_grid)

            kx_grid = float(initial_k0[0]) / K_GRID_FACTOR
            # Flip Y to match the top-down canvas convention
            ky_grid = -float(initial_k0[1]) / K_GRID_FACTOR
            self.canvas.k0_tip_grid = QPointF(rx_grid + kx_grid, ry_grid + ky_grid)

    def _restore_canvas(self, potential_array: np.ndarray) -> None:
        """
        Translates a raw numerical floating-point backend potential matrix back into
        the underlying canvas image. The image is built at native grid resolution
        and no spatial scaling is performed, so reopening the dialog reproduces the
        exact same pixels that were saved.

        Args:
            potential_array (np.ndarray): 2D array (size_x, size_y) representing the
                saved potential landscape.
        """
        # potential_array has shape (size_x, size_y) with row 0 at the canvas top
        # (see save_and_close); transposing yields (size_y, size_x) suitable for
        # QImage's (height, width) row-major layout.
        arr = 255.0 - (potential_array.T / self.initial_wall_height * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        height, width = arr.shape

        # Bring the image to the same grid resolution as the array before swapping in.
        if width != self.canvas.grid_size_x or height != self.canvas.grid_size_y:
            self.canvas.set_grid_size(width, height)

        gray_bytes = arr.tobytes()
        gray_img = QImage(
            gray_bytes,
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        ).copy()
        self.canvas.set_image(gray_img.convertToFormat(QImage.Format.Format_ARGB32))

    def _setup_ui(self) -> None:
        """
        Sets up the visual interface layout dynamically via automated Qt structural managers.
        Uses a side-panel architecture with grouped settings (QGroupBox) for better UX.
        """
        # Main window layout (vertical: top is content, bottom is action buttons)
        main_layout = QVBoxLayout(self)

        # HBoxLayout dividing the window into left (Canvas) and right (Settings) sides
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        # ==========================================
        # LEFT SIDE: Canvas
        # ==========================================
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.canvas_container, stretch=1)
        content_layout.addLayout(left_panel, stretch=3)  # Giving the canvas more space

        # ==========================================
        # RIGHT SIDE: Settings Panel (with ScrollArea)
        # ==========================================
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setSpacing(15)

        # --- GROUP 1: Simulation Settings ---
        sim_group = QGroupBox("Simulation Settings")
        sim_form = QFormLayout()

        self.simulation_menu_desc = "Select the numerical method for time evolution."
        self.simulation_menu_label = QLabel("Simulation Method:")
        self.simulation_menu_label.setToolTip(self.simulation_menu_desc)
        self.simulation_menu = QComboBox()
        self.simulation_menu.addItems(["Crank-Nicolson", "SSFM", "Symmetric SSFM"])
        self.simulation_menu.setCurrentText(self.initial_method)
        self.simulation_menu.currentTextChanged.connect(self.simulation_changed.emit)

        self.frames_input_desc = "The total number of frames calculated."
        self.frames_input_label = QLabel("Total Frames (n<sub>tot</sub>):")
        self.frames_input_label.setToolTip(self.frames_input_desc)
        self.frames_input = QSpinBox()
        self.frames_input.setRange(10, 10000)
        self.frames_input.setValue(self.current_frames)

        self.delta_t_input_desc = "The time step determines <br>the size of each integration step <br>in the simulation. Measured in atomic units [a. u.]: <br>1 a. u. \u2248 2.4188843265864(26) \u00d7 10<sup>-17</sup> s"
        self.delta_t_input_label = QLabel("Time Step (\u03b4t) [a. u.]:")
        self.delta_t_input_label.setToolTip(self.delta_t_input_desc)
        self.delta_t_input = QDoubleSpinBox()
        self.delta_t_input.setDecimals(5)
        self.delta_t_input.setRange(0.00001, 1.0)
        self.delta_t_input.setSingleStep(0.001)
        self.delta_t_input.setValue(self.initial_delta_t)

        self.steps_per_frame_input_desc = (
            "The number of simulation steps calculated in each frame."
        )
        self.steps_per_frame_input_label = QLabel("Steps per Frame (n<sub>step</sub>):")
        self.steps_per_frame_input_label.setToolTip(self.steps_per_frame_input_desc)
        self.steps_per_frame_input = QSpinBox()
        self.steps_per_frame_input.setRange(1, 2000)
        self.steps_per_frame_input.setValue(self.initial_steps_per_frame)

        sim_form.addRow(self.simulation_menu_label, self.simulation_menu)
        sim_form.addRow(self.frames_input_label, self.frames_input)
        sim_form.addRow(self.delta_t_input_label, self.delta_t_input)
        sim_form.addRow(self.steps_per_frame_input_label, self.steps_per_frame_input)
        sim_group.setLayout(sim_form)
        right_panel_layout.addWidget(sim_group)

        # --- GROUP 2: Grid & Domain ---
        grid_group = QGroupBox("Grid Domain")
        grid_form = QFormLayout()

        self.x_limit_input_desc = "The width of the simulation <br>domain. Measured in units <br>of Bohr radii [a<sub>0</sub>], where: <br>1 a<sub>0</sub> \u2248 5.29177210544(82) \u00d7 10<sup>-11</sup> m."
        self.x_limit_input_label = QLabel("Width (w) [a<sub>0</sub>]:")
        self.x_limit_input_label.setToolTip(self.x_limit_input_desc)
        self.x_limit_input = QDoubleSpinBox()
        self.x_limit_input.setRange(1.0, 1000.0)
        self.x_limit_input.setValue(self.x_limit)

        self.y_limit_input_desc = """
            The height of the simulation <br>
            domain. Measured in units <br>
            of Bohr radii [a<sub>0</sub>], where: <br>
            1 a<sub>0</sub> \u2248 5.29177210544(82) \u00d7 10<sup>-11</sup> m."""
        self.y_limit_input_label = QLabel("Height (h) [a<sub>0</sub>]:")
        self.y_limit_input_label.setToolTip(self.y_limit_input_desc)
        self.y_limit_input = QDoubleSpinBox()
        self.y_limit_input.setRange(1.0, 1000.0)
        self.y_limit_input.setValue(self.y_limit)

        self.grid_step_input_desc = """
            The grid step determines the spacing <br>
            between grid points in the simulation. <br>
            Smaller steps provide higher spatial <br>
            resolution but increase computation time."""
        self.grid_step_input_label = QLabel(
            "Grid Step (\u03b4) [a<sub>0</sub>\u207b\u00b9]:"
        )
        self.grid_step_input_label.setToolTip(self.grid_step_input_desc)
        self.grid_step_input = QDoubleSpinBox()
        self.grid_step_input.setDecimals(3)
        self.grid_step_input.setRange(0.01, 10.0)
        self.grid_step_input.setValue(self.initial_grid_step)
        self.grid_step_input.setSingleStep(0.05)
        self.x_limit_input.valueChanged.connect(self.update_canvas_size)
        self.y_limit_input.valueChanged.connect(self.update_canvas_size)
        self.grid_step_input.valueChanged.connect(self.update_canvas_size)

        self.wall_height_input_desc = """
            The wall height parameter sets <br>
            the maximum potential value in the simulation. <br>
            Measured in Hartrees [Ha]: <br>
            1 Ha \u2248 27.2113860243679(50) eV."""
        self.wall_height_input_label = QLabel("Wall Height (V<sub>0</sub>) [Ha]:")
        self.wall_height_input_label.setToolTip(self.wall_height_input_desc)
        self.wall_height_input = QDoubleSpinBox()
        self.wall_height_input.setRange(1.0, 1000000.0)
        self.wall_height_input.setSingleStep(10.0)
        self.wall_height_input.setValue(self.initial_wall_height)

        grid_form.addRow(self.x_limit_input_label, self.x_limit_input)
        grid_form.addRow(self.y_limit_input_label, self.y_limit_input)
        grid_form.addRow(self.grid_step_input_label, self.grid_step_input)
        grid_form.addRow(self.wall_height_input_label, self.wall_height_input)
        grid_group.setLayout(grid_form)
        right_panel_layout.addWidget(grid_group)

        # --- GROUP 3: Wavepacket Parameters ---
        wave_group = QGroupBox("Wavepacket Parameters")
        wave_form = QFormLayout()

        self.mass_input_desc = """
            The mass of the simulated particle, <br>
            expressed in units of the electron mass [m<sub>e</sub>]: <br>
            1 m<sub>e</sub> = 9.1093837139(28) \u00d7 10<sup>-31</sup> kg.
            """
        self.mass_input_label = QLabel("Mass (m) [m<sub>e</sub>]:")
        self.mass_input_label.setToolTip(self.mass_input_desc)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.01, 100.0)
        self.mass_input.setValue(self.initial_mass)
        self.mass_input.setSingleStep(0.1)

        self.sig_xx_input_desc = """
            The uncertainty in the x-direction <br>
            of the initial wavepacket, <br>
            expressed in units of the Bohr radii [a<sub>0</sub>]: <br>
            1 a<sub>0</sub> \u2248 5.29177210544(82) \u00d7 10<sup>-11</sup> m.
            """
        self.sig_xx_input_label = QLabel(
            "&sigma;<sub>xx</sub> [a<sub>0</sub><sup>2</sup>]:"
        )
        self.sig_xx_input_label.setToolTip(self.sig_xx_input_desc)
        self.sig_xx_input = QDoubleSpinBox()
        self.sig_xx_input.setRange(0.1, 50.0)
        self.sig_xx_input.setValue(4.0)
        self.sig_xx_input.setSingleStep(0.1)

        self.sig_xy_input_desc = """
            The correlation between uncertainties <br>
            in the x and y directions, <br>
            expressed in units of the Bohr radii [a<sub>0</sub>]: <br>
            1 a<sub>0</sub> \u2248 5.29177210544(82) \u00d7 10<sup>-11</sup> m.
            """
        self.sig_xy_input_label = QLabel(
            "&sigma;<sub>xy</sub> [a<sub>0</sub><sup>2</sup>]:"
        )
        self.sig_xy_input_label.setToolTip(self.sig_xy_input_desc)
        self.sig_xy_input = QDoubleSpinBox()
        self.sig_xy_input.setRange(-100.0, 100.0)
        self.sig_xy_input.setValue(0.0)
        self.sig_xy_input.setSingleStep(0.1)

        self.sig_yy_input_desc = """
            The uncertainty in the y-direction <br>
            of the initial wavepacket, <br>
            expressed in units of the Bohr radii [a<sub>0</sub>]: <br>
            1 a<sub>0</sub> \u2248 5.29177210544(82) \u00d7 10<sup>-11</sup> m.
            """
        self.sig_yy_input_label = QLabel(
            "&sigma;<sub>yy</sub> [a<sub>0</sub><sup>2</sup>]:"
        )
        self.sig_yy_input_label.setToolTip(self.sig_yy_input_desc)
        self.sig_yy_input = QDoubleSpinBox()
        self.sig_yy_input.setRange(0.1, 50.0)
        self.sig_yy_input.setValue(4.0)
        self.sig_yy_input.setSingleStep(0.1)

        if self.initial_sigma is not None:
            self.sig_xx_input.setValue(float(self.initial_sigma[0, 0]))
            self.sig_xy_input.setValue(float(self.initial_sigma[0, 1]))
            self.sig_yy_input.setValue(float(self.initial_sigma[1, 1]))

        wave_form.addRow(self.mass_input_label, self.mass_input)
        wave_form.addRow(self.sig_xx_input_label, self.sig_xx_input)
        wave_form.addRow(self.sig_xy_input_label, self.sig_xy_input)
        wave_form.addRow(self.sig_yy_input_label, self.sig_yy_input)
        wave_group.setLayout(wave_form)
        right_panel_layout.addWidget(wave_group)

        # --- GROUP 4: Editor Mode ---
        mode_group = QGroupBox("Editor Mode")
        mode_layout = QHBoxLayout()

        self.radio_brush = QRadioButton("Brush")
        self.radio_brush.setToolTip(
            "Select the Brush tool to add potential <br>features to the canvas. <br>"
            "Click and drag on the canvas <br>to paint potential barriers based on <br>"
            "the current brush strength and width settings."
        )
        self.radio_brush.setChecked(True)
        self.radio_eraser = QRadioButton("Erase")
        self.radio_eraser.setToolTip(
            "Select the Erase tool to remove potential <br>"
            "features from the canvas. <br>"
            "Click and drag on the canvas <br>"
            "to erase previously drawn potential barriers, <br>"
            "effectively resetting those areas to zero potential."
        )
        self.radio_wave = QRadioButton("Wavepacket r\u2080 && k\u2080")
        self.radio_wave.setToolTip(
            "Select the Wavepacket tool to set <br>"
            "the initial position (r\u2080) and momentum (k\u2080) <br>"
            "of the wavepacket. <br>"
            "Click on the canvas to place the initial position, <br>"
            "and drag to define the initial momentum vector."
        )

        self.radio_brush.toggled.connect(self.update_mode)
        self.radio_eraser.toggled.connect(self.update_mode)
        self.radio_wave.toggled.connect(self.update_mode)

        mode_layout.addWidget(self.radio_brush)
        mode_layout.addWidget(self.radio_eraser)
        mode_layout.addWidget(self.radio_wave)

        mode_group.setLayout(mode_layout)
        right_panel_layout.addWidget(mode_group)

        # --- GROUP 5: Tools & Potential ---
        tools_group = QGroupBox("Potential Editor Tools (Brush && Erease)")
        tools_layout = QVBoxLayout()

        preset_form = QFormLayout()
        self.preset_menu_desc = """
        Load predefined potential landscapes <br>
        to quickly set up common scenarios. <br>
        Selecting a preset will overwrite the current canvas, <br>
        so use with caution if you have custom <br>
        drawings you wish to keep.
        """
        self.preset_menu_label = QLabel("Load Preset Potential:")
        self.preset_menu_label.setToolTip(self.preset_menu_desc)
        self.preset_menu = QComboBox()
        self.preset_menu.addItems(
            [
                "Custom / Clear",
                "Gaussian Bump",
                "Harmonic Oscillator",
                "W-shape",
                "Matryoshka",
            ]
        )
        self.preset_menu.textActivated.connect(self.load_preset_potential)
        preset_form.addRow(self.preset_menu_label, self.preset_menu)
        tools_layout.addLayout(preset_form)

        # Sliders changed to horizontal (to better fit the side panel)
        slider_form = QFormLayout()

        self.brush_strength_desc = """
        Adjust the strength of the brush tool. <br>
        The value 100 corresponds to the maximum <br>
        potential height defined by the wall height setting, <br>
        while 0 corresponds to no potential.
        """
        self.brush_strength_label = QLabel("Strength: 15")
        self.brush_strength_label.setToolTip(self.brush_strength_desc)
        self.brush_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_strength_slider.setRange(1, 100)
        self.brush_strength_slider.setValue(15)

        self.brush_width_desc = """
        Adjust the width of the brush tool. <br>
        The value corresponds to the diameter of <br>
        the circular brush in grid cells.
        """
        self.brush_width_label = QLabel("Width: 3")
        self.brush_width_label.setToolTip(self.brush_width_desc)
        self.brush_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_width_slider.setRange(1, 30)
        self.brush_width_slider.setValue(3)

        def set_strength(v):
            self.brush_strength_label.setText(f"Strength: {v}")
            self.canvas.brush_strength = v

        def set_width(v):
            self.brush_width_label.setText(f"Width: {v}")
            self.canvas.brush_width = v

        self.brush_strength_slider.valueChanged.connect(set_strength)
        self.brush_width_slider.valueChanged.connect(set_width)

        slider_form.addRow(self.brush_strength_label, self.brush_strength_slider)
        slider_form.addRow(self.brush_width_label, self.brush_width_slider)
        tools_layout.addLayout(slider_form)

        tools_group.setLayout(tools_layout)
        right_panel_layout.addWidget(tools_group)

        right_panel_layout.addStretch()

        # Wrap the right panel in a ScrollArea (protects against clipping on small screens)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(right_panel_widget)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(400)
        content_layout.addWidget(scroll_area)

        # ==========================================
        # BOTTOM BAR: Actions
        # ==========================================
        controls = QHBoxLayout()

        self.load_params_btn = QPushButton("Load from JSON")
        self.load_params_btn.setToolTip(
            "Load simulation parameters and potential from a JSON file. <br>"
            "This will overwrite the current canvas and settings, <br>"
            "so use with caution if you have unsaved custom configurations."
        )
        self.load_params_btn.clicked.connect(self.load_params_from_file)

        self.save_params_btn = QPushButton("Save to JSON")
        self.save_params_btn.setToolTip(
            "Save simulation parameters and potential to a JSON file."
        )
        self.save_params_btn.clicked.connect(self.save_params_to_file)

        clear_btn = QPushButton("Clear Potential")
        clear_btn.clicked.connect(self.clear_canvas)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("Save && Update Simulation")
        # Highlighting the main button
        self.save_btn.setStyleSheet("font-weight: bold; padding: 5px 15px;")
        self.save_btn.clicked.connect(self.save_and_close)

        controls.addWidget(self.load_params_btn)
        controls.addWidget(self.save_params_btn)
        controls.addStretch()
        controls.addWidget(clear_btn)
        controls.addWidget(cancel_btn)
        controls.addWidget(self.save_btn)

        main_layout.addLayout(controls)

        # Increase minimum width of spinboxes for convenience
        for spinbox in self.findChildren((QSpinBox, QDoubleSpinBox)):
            spinbox.setMinimumWidth(80)

        self.check_memory_limit()

    def update_canvas_size(self, _value: Optional[float] = None) -> None:
        """
        Dynamically resizes the drawing canvas to a new grid resolution. The
        underlying QImage is rescaled once here (a deliberate, user-initiated
        interpolation); the aspect-ratio container and stored wavepacket
        anchors are synchronised so on-screen content stays in place.
        """
        new_x_limit = self.x_limit_input.value()
        new_y_limit = self.y_limit_input.value()

        grid_step = self.grid_step_input.value()

        # Calculate and truncate the remainder from division
        new_x = int(new_x_limit / grid_step)
        new_y = int(new_y_limit / grid_step)

        if new_x <= 0 or new_y <= 0:
            return

        self.grid_size_x = new_x
        self.grid_size_y = new_y

        aspect_ratio = new_y / new_x
        self.canvas_container.set_aspect_ratio(aspect_ratio)
        self.canvas.set_grid_size(new_x, new_y)

    def check_memory_limit(self) -> None:
        """
        Calculates the maximum number of frames based on the grid size and available RAM.
        Lowers the frames input if it exceeds the calculated limit and notifies the user.
        """
        try:
            import psutil

            mem_available = psutil.virtual_memory().available
        except ImportError:
            # Fallback if psutil is not available, assume 16GB free memory
            mem_available = 16 * 1024 * 1024 * 1024

        # Reserve 2GB buffer for OS, cache
        safe_mem = max(0, mem_available - 2000 * 1024 * 1024)

        nx = int(self.x_limit_input.value() / self.grid_step_input.value())
        ny = int(self.y_limit_input.value() / self.grid_step_input.value())

        # Conservative estimation:
        # np.complex128 takes up 16 bytes
        bytes_per_frame = nx * ny * 16

        if bytes_per_frame == 0:
            return

        max_frames = int(safe_mem / bytes_per_frame)

        # Clamp to reasonable UI boundaries
        max_frames = min(max_frames, 10000)
        max_frames = max(max_frames, 10)

        current_frames = self.frames_input.value()

        # Update the maximum limit of the spinbox
        self.frames_input.setMaximum(max_frames)

        # Apply the reduction and notify if the current frames exceed the new limit
        if current_frames > max_frames:
            self.frames_input.setValue(max_frames)
            QMessageBox.warning(
                self,
                "Memory Limit Reached",
                f"The grid size is too large for the current number of frames.\n\n"
                f"Based on available RAM, the maximum number of frames has been safely lowered to {max_frames}.",
            )

    def load_preset_potential(self, text: str) -> None:
        """
        Loads a predefined mathematical potential from the backend onto the canvas.
        Replaces the current drawing with the generated matrix and configures optimal wavepacket paths.
        """
        if text == "Custom / Clear":
            self.clear_canvas()
            return

        potential_matrix = None
        wall_val = self.wall_height_input.value()

        if text == "Gaussian Bump":
            # Position the peak right in the middle of the simulated grid space
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            # Symmetric covariance matrix creating a smooth circular hill obstacle
            sigma0 = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float64)
            pot = GaussianBumpPotential(
                self.grid_size_x, self.grid_size_y, r0=r0, V0=wall_val, sigma0=sigma0
            )
            potential_matrix = pot.matrix

        elif text == "Harmonic Oscillator":
            r0 = (self.grid_size_x // 2, self.grid_size_y // 2)
            max_dist_sq = r0[0] ** 2 + r0[1] ** 2
            k = (wall_val * 2) / max_dist_sq if max_dist_sq > 0 else 1.0
            pot = HarmonicPotential(self.grid_size_x, self.grid_size_y, k=k, r0=r0)
            potential_matrix = pot.matrix

        elif text == "W-shape":
            # Custom W-shaped potential in the middle of the grid space
            w_size_x = self.grid_size_x // 2
            w_size_y = self.grid_size_y // 2

            pot = WShaped(w_size_x, w_size_y, thickness=3, wall_value=wall_val)
            w_matrix = pot.matrix

            pos_x = (self.grid_size_x - w_size_x) // 2
            pos_y = (self.grid_size_y - w_size_y) // 2

            zero_pot = np.zeros((self.grid_size_x, self.grid_size_y))
            zero_pot[pos_x : pos_x + w_size_x, pos_y : pos_y + w_size_y] = w_matrix
            potential_matrix = zero_pot

            # Drop the wavepacket near the top centre of the well with downward momentum.
            rx_grid = self.grid_size_x * 0.5
            ry_grid = self.grid_size_y * 0.2
            self.canvas.r0_grid = QPointF(rx_grid, ry_grid)
            # Arrow tip 20% of the canvas height below the anchor (positive Qt-Y =
            # negative physics-Y momentum after the save_and_close Y-flip).
            self.canvas.k0_tip_grid = QPointF(rx_grid, ry_grid + self.grid_size_y * 0.2)
            self.sig_xx_input.setValue(4.0)
            self.sig_yy_input.setValue(4.0)

        elif text == "Matryoshka":
            # Custom embedded potential with a central well and surrounding barriers
            # Calculating the inner potential from previous potential data
            scaled_img = self.canvas.image.scaled(
                self.grid_size_x // 2,
                self.grid_size_y // 2,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            gray_img = scaled_img.convertToFormat(QImage.Format.Format_Grayscale8)
            width, height = gray_img.width(), gray_img.height()
            bpl = gray_img.bytesPerLine()
            buffer = gray_img.constBits().asarray(height * bpl)

            arr = (
                np.frombuffer(bytes(buffer), dtype=np.uint8)
                .reshape((height, bpl))
                .copy()
            )
            arr = arr[:, :width]

            inner_matrix = ((255 - arr) / 255.0 * wall_val).T

            inner_matrix[0, :] = wall_val
            inner_matrix[-1, :] = wall_val
            inner_matrix[:, 0] = wall_val
            inner_matrix[:, -1] = wall_val

            inner_pot_obj = Potential(inner_matrix)

            pos_x = self.grid_size_x // 4
            pos_y = self.grid_size_y // 4

            pot = EmbeddedPotential(
                self.grid_size_x, self.grid_size_y, pos_x, pos_y, inner_pot_obj
            )
            potential_matrix = pot.matrix

        if potential_matrix is not None:
            # Leverage the existing canvas reconstruction pipeline
            self._restore_canvas(potential_matrix)
            self.update()

    def update_mode(self) -> None:
        """
        Updates the drawing state based on the current radio button selection.
        """
        if self.radio_brush.isChecked():
            self.canvas.mode = "brush"
        elif self.radio_eraser.isChecked():
            self.canvas.mode = "eraser"
        else:
            self.canvas.mode = "wavepacket"

    def clear_canvas(self) -> None:
        """
        Clears the drawn potential to a blank white canvas and resets the preset dropdown.
        """
        self.preset_menu.setCurrentText("Custom / Clear")
        self.canvas.image.fill(Qt.GlobalColor.white)
        self.canvas.update()

    def save_and_close(self) -> None:
        """
        Parses canvas drawing and physics inputs, emits them, and closes the dialog.
        """
        self.save_btn.setText("Loading...")
        self.save_btn.setEnabled(False)
        QApplication.processEvents()

        # Capture the newly desired grid sizes from user input
        new_x_limit = self.x_limit_input.value()
        new_y_limit = self.y_limit_input.value()
        grid_step = self.grid_step_input.value()

        new_size_x = int(new_x_limit / grid_step)
        new_size_y = int(new_y_limit / grid_step)
        wall_height = self.wall_height_input.value()

        # 1. Process Potential Matrix
        # The canvas image already lives at its current grid resolution. We only
        # interpolate here if the user explicitly changed the grid size input
        # without pressing "Snap Aspect Ratio" first.
        if (
            self.canvas.grid_size_x != new_size_x
            or self.canvas.grid_size_y != new_size_y
        ):
            img_at_grid = self.canvas.image.scaled(
                new_size_x,
                new_size_y,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            img_at_grid = self.canvas.image

        gray_img = img_at_grid.convertToFormat(QImage.Format.Format_Grayscale8)
        width, height = gray_img.width(), gray_img.height()
        bpl = gray_img.bytesPerLine()
        buffer = gray_img.constBits().asarray(height * bpl)

        arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * wall_height
        potential = potential.T

        # 2. Process Wavepacket Parameters in grid coordinates (no canvas-pixel detour)
        if self.canvas.r0_grid is not None and self.canvas.k0_tip_grid is not None:
            # Match the wavepacket to the requested grid resolution if it changed.
            scale_x = new_size_x / self.canvas.grid_size_x
            scale_y = new_size_y / self.canvas.grid_size_y

            r0x_g = self.canvas.r0_grid.x() * scale_x
            r0y_g = self.canvas.r0_grid.y() * scale_y
            k0x_g = self.canvas.k0_tip_grid.x() * scale_x
            k0y_g = self.canvas.k0_tip_grid.y() * scale_y

            rx = float(np.clip(r0x_g, 0.0, float(new_size_x - 1)))
            # Flip from top-down (Qt) back to bottom-up (physics)
            ry = float(np.clip(new_size_y - r0y_g, 0.0, float(new_size_y - 1)))
            r0 = np.array([rx, ry], dtype=np.float64)

            kx = (k0x_g - r0x_g) * K_GRID_FACTOR
            ky = -(k0y_g - r0y_g) * K_GRID_FACTOR
            k0 = np.array([kx, ky], dtype=np.float64)
        else:
            r0 = np.array([0.0, 0.0], dtype=np.float64)
            k0 = np.array([0.0, 0.0], dtype=np.float64)

        # 3. Process Sigma Matrix and Mass
        sig_xx = self.sig_xx_input.value()
        sig_xy = self.sig_xy_input.value()
        sig_yy = self.sig_yy_input.value()

        # Build the symmetric 2x2 covariance matrix
        sigma_matrix = np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])

        mass = self.mass_input.value()
        frames = self.frames_input.value()

        delta_t = self.delta_t_input.value()
        steps_per_frame = self.steps_per_frame_input.value()
        method_name = self.simulation_menu.currentText()

        # Pre-check solver warnings before closing the drawer so the user can
        # adjust parameters without losing the in-progress setup.
        potential_obj = coarse_potential_from_drawer(potential, wall_height)
        r0_int: tuple[int, int] = (int(r0[0]), int(r0[1]))

        packet_capture = WarningCaptureHandler()
        packet_logger = logging.getLogger("backend.StationaryWaveFunc")
        packet_logger.addHandler(packet_capture)

        try:
            wavefunc = GaussianPacket(
                r0_int,
                k0.copy(),
                sigma_matrix.copy(),
                mass,
                new_size_x,
                new_size_y,
            )
        finally:
            packet_logger.removeHandler(packet_capture)

        if packet_capture.captured_warnings:
            warning_text = "\n\n".join(packet_capture.captured_warnings)
            QMessageBox.warning(
                self,
                "Wavepacket Initialization Error",
                f"Cannot create Gaussian packet with current parameters:\n\n{warning_text}",
            )
            self.save_btn.setText("Save && Update Simulation")
            self.save_btn.setEnabled(True)
            return

        simulation, stability_warnings = instantiate_solver_with_warnings(
            method_name=method_name,
            potential=potential_obj,
            wavefunc=wavefunc,
            delta_t=delta_t,
            grid_step=grid_step,
        )

        if stability_warnings:
            warning_text = "\n\n".join(stability_warnings)
            QMessageBox.warning(
                self,
                "Simulation Stability Warning",
                "The physical parameters might cause the simulation to become unstable "
                f"or mathematically inaccurate:\n\n{warning_text}",
            )
            self.save_btn.setText("Save && Update Simulation")
            self.save_btn.setEnabled(True)
            return

        # Emit all parameters to the main window, including the solver we
        # just built for the stability check so it does not need rebuilding.
        self.setup_saved.emit(
            potential,
            r0,
            k0,
            sigma_matrix,
            mass,
            frames,
            new_size_x,
            new_size_y,
            delta_t,
            steps_per_frame,
            wall_height,
            new_x_limit,
            new_y_limit,
            grid_step,
            method_name,
            simulation,
        )
        self.accept()

    def load_params_from_file(self) -> None:
        """
        Loads simulation parameters from a JSON file and updates the UI.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation Parameters", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        p = Params()
        p.read(file_path)

        # Update pure numeric fields
        self.x_limit_input.setValue(p.x_limit)
        self.y_limit_input.setValue(p.y_limit)
        self.grid_step_input.setValue(p.grid_step)
        self.mass_input.setValue(p.mass)
        self.frames_input.setValue(p.updates_max)

        self.delta_t_input.setValue(p.delta_t)
        self.steps_per_frame_input.setValue(p.delta_n)
        self.wall_height_input.setValue(p.well_height)

        # Update Sigma matrix inputs
        self.sig_xx_input.setValue(p.sigma0[0][0])
        self.sig_xy_input.setValue(p.sigma0[0][1])
        self.sig_yy_input.setValue(p.sigma0[1][1])

        # Map SolverType back to dropdown text
        if p.solver == SolverType.CN:
            self.simulation_menu.setCurrentText("Crank-Nicolson")
        elif p.solver == SolverType.SSFM:
            self.simulation_menu.setCurrentText("SSFM")

        # Map WellType back to dropdown presets
        if p.potential_matrix is not None:
            # Custom matrix loading
            self.preset_menu.setCurrentText("Custom / Clear")
            self._restore_canvas(p.potential_matrix)
        elif p.well_type == WellType.W_SHAPED:
            self.preset_menu.setCurrentText("W-shape")
            self.load_preset_potential("W-shape")
        elif p.well_type == WellType.MATRYOSHKA:
            self.preset_menu.setCurrentText("Matryoshka")
            self.load_preset_potential("Matryoshka")
        else:
            self.preset_menu.setCurrentText("Custom / Clear")
            self.load_preset_potential("Custom / Clear")

        self.update_canvas_size()

        rx_grid = float(p.r0[0])
        # Conversion from bottom-to-top (physics) to top-to-bottom (Qt grid) Y
        ry_grid = float(self.grid_size_y) - float(p.r0[1])
        self.canvas.r0_grid = QPointF(rx_grid, ry_grid)

        kx_grid = float(p.k0[0]) / K_GRID_FACTOR
        # Conversion from bottom-to-top (physics) to top-to-bottom (Qt grid) Y
        ky_grid = -float(p.k0[1]) / K_GRID_FACTOR
        self.canvas.k0_tip_grid = QPointF(rx_grid + kx_grid, ry_grid + ky_grid)

        self.canvas.update()

    def save_params_to_file(self) -> None:
        """
        Saves simulation parameters to a JSON file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Parameters", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        if file_path and not file_path.endswith(".json"):
            file_path += ".json"

        p = Params()
        p.x_limit = self.x_limit_input.value()
        p.y_limit = self.y_limit_input.value()
        p.grid_step = self.grid_step_input.value()
        p.size_x = int(p.x_limit / p.grid_step)
        p.size_y = int(p.y_limit / p.grid_step)
        p.mass = self.mass_input.value()
        p.updates_max = self.frames_input.value()
        p.delta_t = self.delta_t_input.value()
        p.delta_n = self.steps_per_frame_input.value()
        p.well_height = self.wall_height_input.value()

        p.sigma0 = np.array(
            [
                [self.sig_xx_input.value(), self.sig_xy_input.value()],
                [self.sig_xy_input.value(), self.sig_yy_input.value()],
            ],
            dtype=np.float64,
        )

        solver_text = self.simulation_menu.currentText()
        if solver_text == "Crank-Nicolson":
            p.solver = SolverType.CN
        elif solver_text == "SSFM":
            p.solver = SolverType.SSFM
        elif solver_text == "Symmetric SSFM":
            p.solver = SolverType.SYM_SSFM

        preset_text = self.preset_menu.currentText()
        if preset_text == "W-shape":
            p.well_type = WellType.W_SHAPED
        elif preset_text == "Matryoshka":
            p.well_type = WellType.MATRYOSHKA
        else:
            p.well_type = WellType.INFINITE_WELL

        if self.canvas.r0_grid is not None and self.canvas.k0_tip_grid is not None:
            # Match the wavepacket to the requested grid resolution if it changed.
            scale_x = p.size_x / self.canvas.grid_size_x
            scale_y = p.size_y / self.canvas.grid_size_y

            r0x_g = self.canvas.r0_grid.x() * scale_x
            r0y_g = self.canvas.r0_grid.y() * scale_y
            k0x_g = self.canvas.k0_tip_grid.x() * scale_x
            k0y_g = self.canvas.k0_tip_grid.y() * scale_y

            rx = int(np.clip(r0x_g, 0, p.size_x - 1))
            # Flip from top-down (Qt) back to bottom-up (physics) and snap to int.
            ry = int(np.clip(p.size_y - r0y_g, 0, p.size_y - 1))
            p.r0 = (rx, ry)

            kx = float((k0x_g - r0x_g) * K_GRID_FACTOR)
            ky = float(-(k0y_g - r0y_g) * K_GRID_FACTOR)
            p.k0 = np.array([kx, ky], dtype=np.float64)
        else:
            p.r0 = (p.size_x // 2, p.size_y // 2)
            p.k0 = np.array([0.0, 0.0], dtype=np.float64)

        img_at_grid = self.canvas.image

        gray_img = img_at_grid.convertToFormat(QImage.Format.Format_Grayscale8)
        width, height = gray_img.width(), gray_img.height()
        bpl = gray_img.bytesPerLine()
        buffer = gray_img.constBits().asarray(height * bpl)

        arr = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, bpl)).copy()
        arr = arr[:, :width]
        potential = (255 - arr) / 255.0 * p.well_height

        p.potential_matrix = potential.T

        p.write(file_path)
