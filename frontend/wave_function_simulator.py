# ==============================================================================
# ### --- FILE wave_function_simulator.py --- ###
# ==============================================================================
import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.colors import hsv_to_rgb
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import RectBivariateSpline
from setup_drawer import SetupDrawer
from mock_backend import QuantumMockBackend


class WaveFunctionSimulator(QMainWindow):
    """
    Main 3D Application Window.
    """

    def __init__(self, size_x=60, size_y=40, z_potential_offset=5):
        super().__init__()
        self.setWindowTitle("3D Wave Function & Potential Simulation")
        self.resize(950, 750)

        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.size_fine_x = 3 * self.size_coarse_x
        self.size_fine_y = 3 * self.size_coarse_y
        self.total_frames = 150

        self.aspect_ratio = self.size_coarse_y / self.size_coarse_x
        self.x_limit = 5.0
        self.y_limit = 5.0 * self.aspect_ratio

        self.x_coarse = np.linspace(-self.x_limit, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(-self.y_limit, self.y_limit, self.size_coarse_y)

        # Persistent physical states
        self.potential_coarse = np.zeros(
            (self.size_coarse_x, self.size_coarse_y), dtype=float
        )
        self.r0 = np.array([0.0, 0.0])
        self.k0 = np.array([0.0, 0.0])
        self.sigma0 = self.sigma_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.mass = 1.0

        self.wave_frames = []
        self.z_potential_offset = z_potential_offset

        self.backend = QuantumMockBackend(
            self.x_coarse, self.y_coarse, self.total_frames
        )

        self._setup_ui()
        self._setup_mesh_geometry()

        self.calculate_all_frames()
        self.update_simulation(0)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=20, elevation=30, azimuth=45)
        layout.addWidget(self.view)

        controls_layout = QHBoxLayout()

        self.time_label = QLabel("Time: 0")
        controls_layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.update_simulation)
        controls_layout.addWidget(self.slider)

        self.draw_btn = QPushButton("Simulation Setup")
        self.draw_btn.clicked.connect(self.open_setup_drawer)
        controls_layout.addWidget(self.draw_btn)

        layout.addLayout(controls_layout)

        self.wave_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=True)
        self.view.addItem(self.wave_mesh_item)

        self.potential_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=True)
        self.view.addItem(self.potential_mesh_item)

        grid = gl.GLGridItem()
        grid.setSize(10, 10 * self.aspect_ratio, 0)
        grid.setSpacing(1, 1, 0)
        self.view.addItem(grid)

    def _setup_mesh_geometry(self):
        self.x_fine = np.linspace(-self.x_limit, self.x_limit, self.size_fine_x)
        self.y_fine = np.linspace(-self.y_limit, self.y_limit, self.size_fine_y)
        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine, indexing="ij")

        faces = []
        for i in range(self.size_fine_x - 1):
            for j in range(self.size_fine_y - 1):
                p1 = i * self.size_fine_y + j
                p2 = p1 + 1
                p3 = (i + 1) * self.size_fine_y + j
                p4 = p3 + 1
                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])
        self.faces = np.array(faces, dtype=int)

    def calculate_all_frames(self):
        self.wave_frames = []
        psi = np.zeros((self.size_coarse_x, self.size_coarse_y), dtype=complex)

        for t in range(self.total_frames):
            psi = self.backend.get_frame(
                psi,
                t,
                self.potential_coarse,
                self.r0,
                self.k0,
                self.sigma_matrix,
                self.mass,
            )
            self.wave_frames.append(psi)

    def open_setup_drawer(self):
        drawer = SetupDrawer(
            grid_size_x=self.size_coarse_x,
            grid_size_y=self.size_coarse_y,
            x_limit=self.x_limit,
            y_limit=self.y_limit,
            parent=self,
        )
        drawer.setup_saved.connect(self.apply_setup)
        drawer.exec()

    def apply_setup(self, potential_array, r0, k0, sigma_matrix, mass):
        # Update simulation states
        self.potential_coarse = potential_array
        self.r0 = r0
        self.k0 = k0
        self.sigma_matrix = sigma_matrix

        print(f"Received Setup -> r0: {r0}, k0: {k0}, sigma_matrix: {sigma_matrix}")
        print(
            f"Received potential of dimensions: {len(potential_array)} x {len(potential_array[0])} "
        )

        self.calculate_all_frames()

        spline = RectBivariateSpline(
            self.x_coarse, self.y_coarse, self.potential_coarse
        )
        potential_fine = spline(self.x_fine, self.y_fine)

        Z_potential = -potential_fine * 2.0
        base_gray = 0.7
        gray_values = base_gray - (potential_fine * 0.5)

        rgba = np.zeros((self.size_fine_x * self.size_fine_y, 4))
        rgba[:, 0] = gray_values.reshape(-1)
        rgba[:, 1] = gray_values.reshape(-1)
        rgba[:, 2] = gray_values.reshape(-1)
        rgba[:, 3] = 0.9

        verts = np.column_stack(
            (
                self.X_fine.reshape(-1),
                self.Y_fine.reshape(-1),
                Z_potential.reshape(-1) - self.z_potential_offset,
            )
        )

        mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
        self.potential_mesh_item.setMeshData(meshdata=mesh_data)

        self.update_simulation(self.slider.value())

    def update_simulation(self, frame_idx):
        self.time_label.setText(f"Time: {frame_idx}")
        if not self.wave_frames:
            return

        psi_coarse = self.wave_frames[frame_idx]

        spline_real = RectBivariateSpline(self.x_coarse, self.y_coarse, psi_coarse.real)
        spline_imag = RectBivariateSpline(self.x_coarse, self.y_coarse, psi_coarse.imag)

        psi_interp = spline_real(self.x_fine, self.y_fine) + 1j * spline_imag(
            self.x_fine, self.y_fine
        )
        prob = np.abs(psi_interp) ** 2

        Z_fine = prob * 4.0

        phase = np.angle(psi_interp)
        hue = (phase + np.pi) / (2 * np.pi)
        saturation = np.ones_like(hue)
        value = np.clip(prob * 2.0 + 0.2, 0, 1)

        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)
        rgba = np.dstack((rgb, np.ones_like(hue)))

        verts = np.column_stack(
            (self.X_fine.reshape(-1), self.Y_fine.reshape(-1), Z_fine.reshape(-1))
        )

        mesh_data = gl.MeshData(
            vertexes=verts, faces=self.faces, vertexColors=rgba.reshape(-1, 4)
        )
        self.wave_mesh_item.setMeshData(meshdata=mesh_data)
