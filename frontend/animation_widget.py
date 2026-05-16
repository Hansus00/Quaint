# ==============================================================================
# ### --- FILE animation_widget.py --- ###
# ==============================================================================

from typing import Optional, Any
import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.colors import hsv_to_rgb
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from scipy.interpolate import RectBivariateSpline


class AnimationWidget(QWidget):
    """
    Widget handling the 3D OpenGL rendering of the simulation.
    Takes coarse physics data, interpolates it for a smoother look, 
    and renders it using OpenGL meshes.
    """

    def __init__(
        self,
        size_coarse_x: int,
        size_coarse_y: int,
        x_limit: float,
        y_limit: float,
        z_potential_offset: int = 5,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the 3D rendering environment.

        Args:
            size_coarse_x (int): Coarse physics grid X dimension.
            size_coarse_y (int): Coarse physics grid Y dimension.
            x_limit (float): Physical limit of X axis.
            y_limit (float): Physical limit of Y axis.
            z_potential_offset (int): Offset below Z=0 to render the potential floor.
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.size_coarse_x: int = size_coarse_x
        self.size_coarse_y: int = size_coarse_y
        
        # Scaling factor for smoother visual mesh rendering
        self.size_fine_x: int = 3 * self.size_coarse_x
        self.size_fine_y: int = 3 * self.size_coarse_y

        self.x_limit: float = x_limit
        self.y_limit: float = y_limit
        self.z_potential_offset: float = z_potential_offset

        self.x_coarse: np.ndarray = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse: np.ndarray = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        self._setup_ui()
        self._setup_mesh_geometry()

    def _setup_ui(self) -> None:
        """Sets up the OpenGL View and basic scene objects (axes, grid)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()

        # Center the camera exactly at the middle of the physical coordinate bounds
        self.view.setCameraPosition(
            pos=QVector3D(self.x_limit / 2.0, self.y_limit / 2.0, 0.0),
            distance=20,
            elevation=30,
            azimuth=45,
        )
        layout.addWidget(self.view)

        # Add 3D Coordinate Axes (X=Red, Y=Green, Z=Blue)
        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=self.x_limit, y=self.y_limit, z=5.0)
        self.view.addItem(self.axis)

        self.wave_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=True)
        self.view.addItem(self.wave_mesh_item)

        self.potential_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=True)
        self.view.addItem(self.potential_mesh_item)

        grid = gl.GLGridItem()
        grid.setSize(self.x_limit, self.y_limit, 0)
        grid.setSpacing(1, 1, 0)
        grid.translate(self.x_limit / 2.0, self.y_limit / 2.0, 0)
        self.view.addItem(grid)

    def _setup_mesh_geometry(self) -> None:
        """Pre-calculates the geometry (vertices and faces) for the smooth interpolation meshes."""
        self.x_fine = np.linspace(0.0, self.x_limit, self.size_fine_x)
        self.y_fine = np.linspace(0.0, self.y_limit, self.size_fine_y)
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

    def update_potential(self, potential_coarse: np.ndarray) -> None:
        """
        Updates the 3D potential mesh landscape.

        Args:
            potential_coarse (np.ndarray): 2D array of the coarse potential values.
        """
        spline = RectBivariateSpline(self.x_coarse, self.y_coarse, potential_coarse)
        potential_fine = spline(self.x_fine, self.y_fine)

        # Scale down the physical potential so it fits the visual Z-axis
        visual_scale_factor: float = 3.5 / 50.0 
        Z_potential = potential_fine * visual_scale_factor

        base_gray: float = 0.7
        gray_values = base_gray - ((potential_fine / 50.0) * 0.5)
        gray_values = np.clip(gray_values, 0, 1)

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

    def update_wave(self, psi_coarse: Any) -> None:
        """
        Updates the 3D wave function mesh. 
        Calculates probability (Z height) and phase (HSV Hue).

        Args:
            psi_coarse (StationaryWaveFunc): Complex wave packet state from backend.
        """
        wave_matrix = psi_coarse.matrix
        spline_real = RectBivariateSpline(self.x_coarse, self.y_coarse, wave_matrix.real)
        spline_imag = RectBivariateSpline(self.x_coarse, self.y_coarse, wave_matrix.imag)

        psi_interp = spline_real(self.x_fine, self.y_fine) + 1j * spline_imag(self.x_fine, self.y_fine)
        prob = np.abs(psi_interp) ** 2

        # Exaggerate height to make spreading wave visible
        Z_fine = prob * 15.0

        phase = np.angle(psi_interp)
        hue = (phase + np.pi) / (2 * np.pi)
        saturation = np.ones_like(hue)
        
        value = np.clip(prob, 0, 1)

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