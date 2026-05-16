# ==============================================================================
# ### --- FILE frontend/animation_widget.py --- ###
# ==============================================================================

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom


class AnimationWidget(QWidget):
    """
    Widget handling the 3D OpenGL rendering of the simulation.
    """

    def __init__(
        self,
        size_coarse_x: int,
        size_coarse_y: int,
        x_limit: float,
        y_limit: float,
        z_potential_offset: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.size_coarse_x: int = size_coarse_x
        self.size_coarse_y: int = size_coarse_y

        self.size_fine_x: int = 4 * self.size_coarse_x
        self.size_fine_y: int = 4 * self.size_coarse_y

        self.x_limit: float = x_limit
        self.y_limit: float = y_limit
        self.z_potential_offset: float = z_potential_offset

        self.x_coarse: np.ndarray = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse: np.ndarray = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        # Lazy memory cache mapping frame object IDs -> pre-computed (verts, rgba)
        self._wave_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._setup_ui()
        self._setup_mesh_geometry()

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
        self.view.addItem(self.potential_mesh_item)

        grid = gl.GLGridItem()
        grid.setSize(self.x_limit, self.y_limit, 0)
        grid.setSpacing(1, 1, 0)
        grid.translate(self.x_limit / 2.0, self.y_limit / 2.0, 0)
        self.view.addItem(grid)

    def _setup_mesh_geometry(self) -> None:
        """Pre-calculates static coordinates to eliminate runtime memory allocations."""
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

        # Allocate static X and Y layout template
        self.verts_template = np.zeros(
            (self.size_fine_x * self.size_fine_y, 3), dtype=np.float32
        )
        self.verts_template[:, 0] = self.X_fine.ravel()
        self.verts_template[:, 1] = self.Y_fine.ravel()

    def update_potential(self, potential_coarse: np.ndarray) -> None:
        """Updates the 3D potential landscape and evicts outdated cache tracking."""
        self.clear_cache()

        spline = RectBivariateSpline(self.x_coarse, self.y_coarse, potential_coarse)
        potential_fine = spline(self.x_fine, self.y_fine)

        visual_scale_factor: float = 3.5 / 50.0
        Z_potential = potential_fine * visual_scale_factor

        base_gray: float = 0.7
        gray_values = base_gray - ((potential_fine / 50.0) * 0.5)
        gray_values = np.clip(gray_values, 0, 1)

        rgba = np.zeros((self.size_fine_x * self.size_fine_y, 4), dtype=np.float32)
        rgba[:, 0] = gray_values.reshape(-1)
        rgba[:, 1] = gray_values.reshape(-1)
        rgba[:, 2] = gray_values.reshape(-1)
        rgba[:, 3] = 0.9

        verts = self.verts_template.copy()
        verts[:, 2] = Z_potential.reshape(-1) - self.z_potential_offset

        mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
        self.potential_mesh_item.setMeshData(meshdata=mesh_data)

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

    def update_wave(self, psi_coarse: Any) -> None:
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

        # 1. Smoothly interpolate the probability envelope directly.
        # This eliminates "holes" because the envelope doesn't oscillate.
        prob_coarse = np.abs(wave_matrix) ** 2
        prob_fine = zoom(prob_coarse, (zoom_factor_x, zoom_factor_y), order=3)

        # High-order splines can occasionally dip slightly below 0 at the extreme tail ends,
        # so we clip it to ensure valid physical probability heights.
        prob_fine = np.clip(prob_fine, 0.0, None)
        Z_fine = prob_fine * 15.0

        # 2. Use fast bilinear interpolation (order=1) purely to map the phase colors.
        # Bilinear prevents the phase from "ringing" or overshooting.
        psi_real_linear = zoom(
            wave_matrix.real, (zoom_factor_x, zoom_factor_y), order=1
        )
        psi_imag_linear = zoom(
            wave_matrix.imag, (zoom_factor_x, zoom_factor_y), order=1
        )

        phase = np.angle(psi_real_linear + 1j * psi_imag_linear)
        hue = (phase + np.pi) / (2 * np.pi)
        value = np.clip(prob_fine * 50, 0.0, 1.0)

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
