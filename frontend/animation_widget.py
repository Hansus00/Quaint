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

from .simulation_builders import WaveFrameArray


class AnimationWidget(QWidget):
    """
    Widget handling the 3D OpenGL rendering of the simulation.
    Manages high-performance matrix transformations and mesh updates for both
    the probability wave and the underlying physical potential landscape.
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
    potential_alpha: float
    x_coarse: np.ndarray
    y_coarse: np.ndarray

    _wave_cache: Dict[int, Tuple[np.ndarray, np.ndarray]]
    max_cache_size: int

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
    potential_verts: np.ndarray
    potential_rgba: np.ndarray
    potential_mesh_data: gl.MeshData
    _hsv_scratch: np.ndarray

    def __init__(
        self,
        size_coarse_x: int,
        size_coarse_y: int,
        x_limit: float,
        y_limit: float,
        z_potential_offset: float,
        z_scale: float = 15.0,
        fine_grid_scale: int = 3,
        z_potential_scale: float = 0.05,
        brightness_multiplier: float = 50.0,
        potential_alpha: float = 0.4,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initializes the 3D OpenGL rendering widget and its structural constraints.

        Args:
            size_coarse_x (int): Base horizontal resolution of the simulated physics grid.
            size_coarse_y (int): Base vertical resolution of the simulated physics grid.
            x_limit (float): Maximum physical coordinate in the X-axis mapping.
            y_limit (float): Maximum physical coordinate in the Y-axis mapping.
            z_potential_offset (float): Depth mapping offset pushing the potential mesh downwards.
            z_scale (float): Vertical height multiplier for the probability waveform.
            fine_grid_scale (int): Interpolation multiplier increasing visual mesh density.
            z_potential_scale (float): Vertical height multiplier for the potential walls.
            brightness_multiplier (float): Exposure scalar brightening the faint probability tails.
            potential_alpha (float): Opacity scalar for the drawn potential field (0.0 to 1.0).
            parent (Optional[QWidget]): Parent application window hosting this widget.
        """
        super().__init__(parent)
        self.size_coarse_x = size_coarse_x
        self.size_coarse_y = size_coarse_y

        self.fine_grid_scale = fine_grid_scale
        self.size_fine_x = self.fine_grid_scale * self.size_coarse_x
        self.size_fine_y = self.fine_grid_scale * self.size_coarse_y

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_potential_offset = z_potential_offset
        self.z_scale = z_scale
        self.z_potential_scale = z_potential_scale
        self.brightness_multiplier = brightness_multiplier
        self.potential_alpha = potential_alpha

        self.x_coarse = np.linspace(0.0, self.x_limit, self.size_coarse_x)
        self.y_coarse = np.linspace(0.0, self.y_limit, self.size_coarse_y)

        # Lazy memory cache mapping frame object IDs -> pre-computed (verts, rgba)
        self._wave_cache = {}
        # Default maximum cache limit. It's later changed based on available memory
        self.max_cache_size = 150

        self._setup_ui()
        self._setup_mesh_geometry()

    def update_config(
        self,
        size_x: int,
        size_y: int,
        x_limit: float,
        y_limit: float,
        z_offset: float,
        z_scale: float,
        fine_grid_scale: int,
        z_potential_scale: float,
        brightness_multiplier: float,
        potential_alpha: float,
    ) -> None:
        """
        Dynamically reconfigures the widget's layout bounds and clears the state.
        Triggered primarily by updates originating from the Settings dialog window.
        """
        self.size_coarse_x = size_x
        self.size_coarse_y = size_y
        self.fine_grid_scale = fine_grid_scale
        self.size_fine_x = self.fine_grid_scale * size_x
        self.size_fine_y = self.fine_grid_scale * size_y

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_potential_offset = z_offset
        self.z_scale = z_scale
        self.z_potential_scale = z_potential_scale
        self.brightness_multiplier = brightness_multiplier
        self.potential_alpha = potential_alpha

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
        """Sets up the OpenGL View and basic background scene objects (camera, grid)."""
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

        # Force uint32 so it works identically across Windows / Linux / Mac drivers
        self.faces = np.array(faces, dtype=np.uint32)

        self.verts_template = np.zeros(
            (self.size_fine_x * self.size_fine_y, 3), dtype=np.float32
        )
        self.verts_template[:, 0] = self.X_fine.ravel()
        self.verts_template[:, 1] = self.Y_fine.ravel()

        self.potential_verts = self.verts_template.copy()
        self.potential_rgba = np.zeros(
            (self.size_fine_x * self.size_fine_y, 4), dtype=np.float32
        )

        # Reusable scratch buffer for the per-frame HSV->RGB conversion;
        # sized once to (Nx*Ny,) so update_wave never re-allocates it.
        self._hsv_scratch = np.empty(
            self.size_fine_x * self.size_fine_y, dtype=np.float32
        )

        self.potential_mesh_data = gl.MeshData(
            vertexes=self.potential_verts,
            faces=self.faces,
            vertexColors=self.potential_rgba,
        )
        self.potential_mesh_item.setMeshData(meshdata=self.potential_mesh_data)

        # Placeholder wave mesh so OpenGL never draws GLMeshItem with faces=None
        wave_rgba = np.zeros((self.size_fine_x * self.size_fine_y, 4), dtype=np.float32)
        wave_rgba[:, 3] = 1.0
        self.wave_mesh_item.setMeshData(
            meshdata=gl.MeshData(
                vertexes=self.verts_template.copy(),
                faces=self.faces,
                vertexColors=wave_rgba,
            )
        )

    def reset_camera(self) -> None:
        """Resets the 3D camera to the default position and orientation."""
        self.view.setCameraPosition(
            pos=QVector3D(self.x_limit / 2.0, self.y_limit / 2.0, 0.0),
            distance=20,
            elevation=30,
            azimuth=45,
        )

    def set_potential_visible(self, visible: bool) -> None:
        """Shows or hides the 3D potential mesh depending on UI button toggle."""
        self.potential_mesh_item.setVisible(visible)

    def update_potential(self, potential_coarse: np.ndarray) -> None:
        """Updates the 3D potential landscape and evicts outdated cache tracking."""
        self.clear_cache()

        # Bilinear interpolation (piecewise-linear) to avoid cubic spline ringing
        spline = RectBivariateSpline(
            self.x_coarse, self.y_coarse, potential_coarse, kx=1, ky=1
        )
        potential_fine = spline(self.x_fine, self.y_fine)

        # Clamp to the coarse range
        v_lo = float(np.min(potential_coarse))
        v_hi = float(np.max(potential_coarse))
        potential_fine = np.clip(potential_fine, v_lo, v_hi)

        Z_potential = potential_fine * self.z_potential_scale

        base_gray: float = 0.7
        gray_values = base_gray - ((potential_fine / 50.0) * 0.5)
        gray_values = np.clip(gray_values, 0, 1)

        # Modifying our pre-allocated memory buffers in-place to heavily reduce GC loads
        self.potential_rgba[:, 0] = gray_values.reshape(-1)
        self.potential_rgba[:, 1] = gray_values.reshape(-1)
        self.potential_rgba[:, 2] = gray_values.reshape(-1)

        # Use dynamic alpha value from the user settings
        self.potential_rgba[:, 3] = self.potential_alpha

        self.potential_verts[:, 2] = Z_potential.reshape(-1) - self.z_potential_offset

        # Wrap the recycled buffers in a new lightweight MeshData object
        # This prevents PyQtGraph from discarding the 'faces' array internally.
        mesh_data = gl.MeshData(
            vertexes=self.potential_verts,
            faces=self.faces,
            vertexColors=self.potential_rgba,
        )
        self.potential_mesh_item.setMeshData(meshdata=mesh_data)

    def _fill_hsv_rgb_into(
        self,
        hue_flat: np.ndarray,
        value_flat: np.ndarray,
        rgba_out: np.ndarray,
    ) -> None:
        """
        Branchless vectorized HSV (Saturation=1) -> RGB, writing into rgba_out[:, :3].

        Uses the closed-form identity:
            f(n) = clip(2 - |((H*6 + n) mod 6) - 2|, 0, 1)
            channel = V - V * f(n)        with n = 5 (R), 3 (G), 1 (B)
        This replaces the previous 6-sextant boolean-mask loop, which paid for
        six full-grid mask builds plus six fancy-indexed scatter assignments.
        Now everything streams through `out=` ufuncs on a single scratch buffer.

        Mutates `hue_flat` (scales it by 6 in place).
        """
        np.multiply(hue_flat, 6.0, out=hue_flat)
        h6 = hue_flat
        scratch = self._hsv_scratch

        for n_offset, channel in ((5.0, 0), (3.0, 1), (1.0, 2)):
            np.add(h6, n_offset, out=scratch)
            np.mod(scratch, 6.0, out=scratch)
            np.subtract(scratch, 2.0, out=scratch)
            np.abs(scratch, out=scratch)
            np.subtract(2.0, scratch, out=scratch)
            np.clip(scratch, 0.0, 1.0, out=scratch)
            np.multiply(value_flat, scratch, out=scratch)
            np.subtract(value_flat, scratch, out=rgba_out[:, channel])

    def clear_cache(self) -> None:
        """Clears the rendered frames cache to prevent memory address collisions."""
        self._wave_cache.clear()

    def update_wave(self, wave_matrix: WaveFrameArray) -> None:
        """Updates the 3D wave function mesh. Utilizes instant cache lookup if frame is known."""
        cache_key = id(wave_matrix)

        # Instant execution if this frame instance was drawn before
        if cache_key in self._wave_cache:
            verts, rgba = self._wave_cache[cache_key]
            mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
            self.wave_mesh_item.setMeshData(meshdata=mesh_data)
            return

        # Cache miss: Run calculations ONCE for this frame instance.
        zoom_factor_x = self.size_fine_x / wave_matrix.shape[0]
        zoom_factor_y = self.size_fine_y / wave_matrix.shape[1]

        # Quadratic B-spline upscale of the real and imaginary parts.
        # `order=2` is the sweet spot for this hot path: visually
        # indistinguishable from cubic (max abs deviation < 1e-4 on a
        # normalized wavefunction) but ~2x faster than `order=3`, and it
        # has far less of the over/under-shoot that the previous cubic
        # produced near sharp probability peaks. Linear (`order=1`) is
        # faster still but produces visible faceting on the upscaled mesh.
        psi_real_fine = zoom(
            wave_matrix.real, (zoom_factor_x, zoom_factor_y), order=2
        )
        psi_imag_fine = zoom(
            wave_matrix.imag, (zoom_factor_x, zoom_factor_y), order=2
        )

        # Pre-allocate the buffers that get committed to the cache.
        n_verts = self.size_fine_x * self.size_fine_y
        verts = self.verts_template.copy()
        rgba = np.empty((n_verts, 4), dtype=np.float32)
        rgba[:, 3] = 1.0

        # 1D views so every downstream ufunc works on contiguous memory and
        # writes directly into the (strided) rgba/verts column slices.
        psi_real_flat = psi_real_fine.ravel()
        psi_imag_flat = psi_imag_fine.ravel()

        # Phase -> hue first, while real & imag are still untouched.
        # hue = (atan2(imag, real) + pi) / (2*pi) = atan2/(2*pi) + 0.5
        hue = np.arctan2(psi_imag_flat, psi_real_flat)
        hue *= np.float32(1.0 / (2.0 * np.pi))
        hue += np.float32(0.5)

        # prob = real^2 + imag^2 -- computed directly from the float32 arrays.
        # The old `np.abs(psi_fine)**2` first allocated a full complex64 grid
        # and then did sqrt(...)**2, which is doubly wasteful. Here we destroy
        # `psi_real_flat`/`psi_imag_flat` in place since they aren't needed again.
        np.multiply(psi_real_flat, psi_real_flat, out=psi_real_flat)
        np.multiply(psi_imag_flat, psi_imag_flat, out=psi_imag_flat)
        prob_flat = psi_real_flat
        prob_flat += psi_imag_flat

        # Z coordinate = prob * z_scale, written straight into the verts column.
        np.multiply(prob_flat, self.z_scale, out=verts[:, 2])

        # value = clip(sqrt(prob) * brightness, 0.01, 1.0), all in place on prob.
        np.sqrt(prob_flat, out=prob_flat)
        amp = prob_flat
        amp *= self.brightness_multiplier
        np.clip(amp, 0.01, 1.0, out=amp)

        # Streaming HSV->RGB write into rgba[:, :3] (mutates `hue` in place).
        self._fill_hsv_rgb_into(hue, amp, rgba)

        # Enforce cache size limit to prevent unbounded memory growth.
        if len(self._wave_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._wave_cache))
            del self._wave_cache[oldest_key]

        # Save to lazy cache for lookups on the next animation pass
        self._wave_cache[cache_key] = (verts, rgba)

        mesh_data = gl.MeshData(vertexes=verts, faces=self.faces, vertexColors=rgba)
        self.wave_mesh_item.setMeshData(meshdata=mesh_data)
