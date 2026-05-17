# ==============================================================================
# ### --- FILE frontend/animation_widget.py --- ###
# ==============================================================================

from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
from backend.StationaryWaveFunc import StationaryWaveFunc


class AnimationWidget(QWidget):
    """
    Widget handling the 3D OpenGL rendering of the simulation.
    """

    # --- Class Fields ---
    size_coarse_x: int
    size_coarse_y: int
    fine_grid_scale: int
    size_fine_x: int
    size_fine_y: int
    x_limit: float
    y_limit: float
    z_potential_offset: float
    z_scale: float
    z_potential_scale: float
    brightness_multiplier: float
    x_coarse: np.ndarray
    y_coarse: np.ndarray
    _wave_cache: Dict[int, Tuple[np.ndarray, np.ndarray]]
    view: gl.GLViewWidget
    axis: gl.GLAxisItem
    wave_mesh_item: gl.GLMeshItem
    potential_mesh_item: gl.GLMeshItem
    grid: gl.GLGridItem
    x_fine: np.ndarray
    y_fine: np.ndarray
    X_fine: np.ndarray
    Y_fine: np.ndarray
    faces: np.ndarray
    verts_template: np.ndarray

    def __init__(
        self,
        size_coarse_x: int,
        size_coarse_y: int,
        x_limit: float,
        y_limit: float,
        z_potential_offset: float,
        z_scale: float = 15.0,
        fine_grid_scale: int = 4,
        z_potential_scale: float = 0.05,
        brightness_multiplier: float = 50.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.size_coarse_x: int = size_coarse_x
        self.size_coarse_y: int = size_coarse_y

        self.fine_grid_scale: int = fine_grid_scale
        self.size_fine_x: int = self.fine_grid_scale * self.size_coarse_x
        self.size_fine_y: int = self.fine_grid_scale * self.size_coarse_y

        self.x_limit: float = x_limit
        self.y_limit: float = y_limit
        self.z_potential_offset: float = z_potential_offset
        self.z_scale: float = z_scale
        self.z_potential_scale: float = z_potential_scale
        self.brightness_multiplier: float = brightness_multiplier

        self.x_coarse: np.ndarray = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse: np.ndarray = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        # Lazy memory cache mapping frame object IDs -> pre-computed (verts, rgba)
        self._wave_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._setup_ui()
        self._setup_mesh_geometry()

    def update_config(
        self,
        size_x: int,
        size_y: int,
        y_limit: float,
        z_offset: float,
        z_scale: float,
        fine_grid_scale: int,
        z_potential_scale: float,
        brightness_multiplier: float,
    ) -> None:
        """Dynamically reconfigures the widget's layout bounds and clears state."""
        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.fine_grid_scale = fine_grid_scale
        self.size_fine_x = self.fine_grid_scale * size_x
        self.size_fine_y = self.fine_grid_scale * size_y

        self.y_limit = y_limit
        self.z_potential_offset = z_offset
        self.z_scale = z_scale
        self.z_potential_scale = z_potential_scale
        self.brightness_multiplier = brightness_multiplier

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        # Re-allocate mesh matrices for the new array dimensions
        self._setup_mesh_geometry()

        # Adjust OpenGL visual bounds
        self.axis.setSize(x=self.x_limit, y=self.y_limit, z=5.0)

        self.view.removeItem(self.grid)
        self.grid = gl.GLGridItem()
        self.grid.setSize(self.x_limit, self.y_limit, 0)
        self.grid.setSpacing(1, 1, 0)
        self.grid.translate(self.x_limit / 2.0, self.y_limit / 2.0, 0)
        self.view.addItem(self.grid)

        self.clear_cache()

    def _setup_ui(self) -> None:
        """Sets up the OpenGL View and basic scene objects."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(
            pos=QVector3D(self.x_limit / 2.0, self.y_limit / 2.0, 0.0),
            distance=20,
            elevation=30,
            azimuth=45,
        )
        layout.addWidget(self.view)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=self.x_limit, y=self.y_limit, z=5.0)
        self.view.addItem(self.axis)

        self.wave_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=False)
        self.view.addItem(self.wave_mesh_item)

        self.potential_mesh_item = gl.GLMeshItem(smooth=True, computeNormals=False)
        self.potential_mesh_item.setGLOptions("translucent")
        self.view.addItem(self.potential_mesh_item)

        self.grid = gl.GLGridItem()
        self.grid.setSize(self.x_limit, self.y_limit, 0)
        self.grid.setSpacing(1, 1, 0)
        self.grid.translate(self.x_limit / 2.0, self.y_limit / 2.0, 0)
        self.view.addItem(self.grid)

    def _setup_mesh_geometry(self) -> None:
        """Pre-calculates static coordinates to eliminate runtime memory allocations."""
        self.x_fine = np.linspace(0.0, self.x_limit, self.size_fine_x)
        self.y_fine = np.linspace(0.0, self.y_limit, self.size_fine_y)
        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine, indexing="ij")

        Nx = self.size_fine_x
        Ny = self.size_fine_y

        I, J = np.meshgrid(np.arange(Nx - 1), np.arange(Ny - 1), indexing="ij")  # noqa: E741

        P1 = I * Ny + J
        P2 = P1 + 1
        P3 = (I + 1) * Ny + J
        P4 = P3 + 1

        triangles1 = np.stack((P1, P2, P3), axis=-1).reshape(-1, 3)
        triangles2 = np.stack((P2, P4, P3), axis=-1).reshape(-1, 3)

        faces = np.empty((P1.size * 2, 3), dtype=np.uint32)
        faces[0::2] = triangles1
        faces[1::2] = triangles2

        # Force uint32 so it works on Linux / Mac
        self.faces = np.array(faces, dtype=np.uint32)

        self.verts_template = np.zeros(
            (self.size_fine_x * self.size_fine_y, 3), dtype=np.float32
        )
        self.verts_template[:, 0] = self.X_fine.ravel()
        self.verts_template[:, 1] = self.Y_fine.ravel()

        self.potential_verts = self.verts_template.copy()
        self.potential_rgba = np.zeros((self.size_fine_x * self.size_fine_y, 4), dtype=np.float32)
        
        self.potential_mesh_data = gl.MeshData(
            vertexes=self.potential_verts, 
            faces=self.faces, 
            vertexColors=self.potential_rgba
        )
        self.potential_mesh_item.setMeshData(meshdata=self.potential_mesh_data)


    def set_potential_visible(self, visible: bool) -> None:
        """Shows or hides the 3D potential mesh."""
        self.potential_mesh_item.setVisible(visible)

    def update_potential(self, potential_coarse: np.ndarray) -> None:
        """Updates the 3D potential landscape"""
        self.clear_cache()

        spline = RectBivariateSpline(self.x_coarse, self.y_coarse, potential_coarse)
        potential_fine = spline(self.x_fine, self.y_fine)

        Z_potential = potential_fine * self.z_potential_scale

        base_gray: float = 0.7
        gray_values = base_gray - ((potential_fine / 50.0) * 0.5)
        gray_values = np.clip(gray_values, 0, 1)

        # 1. Bezpośrednio nadpisujemy naszą "recyklingowaną" macierz kolorów (modyfikacja In-place)
        self.potential_rgba[:, 0] = gray_values.reshape(-1)
        self.potential_rgba[:, 1] = gray_values.reshape(-1)
        self.potential_rgba[:, 2] = gray_values.reshape(-1)
        self.potential_rgba[:, 3] = 0.4  # Alpha

        # 2. Bezpośrednio nadpisujemy wierzchołki
        self.potential_verts[:, 2] = Z_potential.reshape(-1) - self.z_potential_offset

        # 3. Wstrzykujemy nowe dane do ISTNIEJĄCEGO obiektu (zero alokacji nowej pamięci!)
        self.potential_mesh_data.setVertexes(self.potential_verts)
        self.potential_mesh_data.setVertexColors(self.potential_rgba)
        
        # 4. Informujemy OpenGL, że dane uległy zmianie i trzeba je narysować od nowa
        self.potential_mesh_item.meshDataChanged()

    def _fast_hsv_to_rgb(self, hue: np.ndarray, value: np.ndarray) -> np.ndarray:
        """High-speed vectorized HSV converter optimized for Saturation=1.0."""
        h6 = hue * 6.0
        i = h6.astype(np.int32) % 6
        f = h6 - np.floor(h6)

        q = value * (1.0 - f)
        t = value * f

        rgb = np.zeros((self.size_fine_x, self.size_fine_y, 3), dtype=np.float32)

        for k in range(6):
            mask = i == k
            if not np.any(mask):
                continue
            if k == 0:
                rgb[mask, 0], rgb[mask, 1] = value[mask], t[mask]
            elif k == 1:
                rgb[mask, 0], rgb[mask, 1] = q[mask], value[mask]
            elif k == 2:
                rgb[mask, 1], rgb[mask, 2] = value[mask], t[mask]
            elif k == 3:
                rgb[mask, 1], rgb[mask, 2] = q[mask], value[mask]
            elif k == 4:
                rgb[mask, 0], rgb[mask, 2] = t[mask], value[mask]
            elif k == 5:
                rgb[mask, 0], rgb[mask, 2] = value[mask], q[mask]

        return rgb

    def clear_cache(self) -> None:
        """Clears the rendered frames cache to prevent memory address collisions."""
        self._wave_cache.clear()

    def update_wave(self, psi_coarse: StationaryWaveFunc) -> None:
        """Updates the 3D wave function mesh. Utilizes instant cache lookup if frame is known."""
        cache_key = id(psi_coarse)

        # Instant execution if this frame instance was drawn before
        if cache_key in self._wave_cache:
            verts, rgba = self._wave_cache[cache_key]
            mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
            self.wave_mesh_item.setMeshData(meshdata=mesh_data)
            return

        # Cache miss: Run calculations ONCE for this frame instance
        wave_matrix = psi_coarse.matrix
        zoom_factor_x = self.size_fine_x / wave_matrix.shape[0]
        zoom_factor_y = self.size_fine_y / wave_matrix.shape[1]

        prob_coarse = np.abs(wave_matrix) ** 2
        prob_fine = zoom(prob_coarse, (zoom_factor_x, zoom_factor_y), order=3)

        prob_fine = np.clip(prob_fine, 0.0, None)

        Z_fine = prob_fine * self.z_scale

        psi_real_linear = zoom(
            wave_matrix.real, (zoom_factor_x, zoom_factor_y), order=1
        )
        psi_imag_linear = zoom(
            wave_matrix.imag, (zoom_factor_x, zoom_factor_y), order=1
        )

        phase = np.angle(psi_real_linear + 1j * psi_imag_linear)
        hue = (phase + np.pi) / (2 * np.pi)

        # Exposure multiplier for visual brightness
        # Boosts the faint probability tails to be visible without exceeding 1.0 and set minimum brightness 0.01
        value = np.clip(np.sqrt(prob_fine) * self.brightness_multiplier, 0.01, 1.0)

        rgb = self._fast_hsv_to_rgb(hue, value)

        verts = self.verts_template.copy()
        verts[:, 2] = Z_fine.ravel()

        rgba = np.empty((self.size_fine_x * self.size_fine_y, 4), dtype=np.float32)
        rgba[:, :3] = rgb.reshape(-1, 3)
        rgba[:, 3] = 1.0

        # Save to lazy cache for lookups on the next animation pass
        self._wave_cache[cache_key] = (verts, rgba)

        mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
        self.wave_mesh_item.setMeshData(meshdata=mesh_data)
