import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from backend.Potential import Potential
from backend.Solver import SSFM, CrankNicolson, SSFMSymmetric
from backend.StationaryWaveFunc import GaussianPacket

from .warning_handler import WarningCaptureHandler

# Single place controlling frontend wave-frame precision.
waveFrameType: TypeAlias = np.complex64
WaveFrameArray: TypeAlias = NDArray[waveFrameType]


def cast_wave_frame(frame: np.ndarray) -> WaveFrameArray:
    """Cast and compact a wave frame to the configured frontend complex dtype."""
    return np.asarray(frame, dtype=waveFrameType, order="C")


def coarse_potential_from_drawer(
    potential_array: np.ndarray, wall_height: float
) -> Potential:
    """Convert drawer potential (UI orientation) to backend coarse matrix."""
    potential_coarse = potential_array[:, ::-1].copy()
    potential_coarse[0, :] = wall_height
    potential_coarse[-1, :] = wall_height
    potential_coarse[:, 0] = wall_height
    potential_coarse[:, -1] = wall_height
    return Potential(potential_coarse)


def instantiate_solver_with_warnings(
    method_name: str,
    potential: Potential,
    wavefunc: GaussianPacket,
    delta_t: float,
    grid_step: float,
) -> tuple[Any, list[str]]:
    """Instantiate the selected solver and return it alongside captured stability warnings.

    Returns a ``(solver, warnings)`` tuple. The warnings are kept separate from
    the solver object so we don't need to attach a non-standard attribute to
    the backend `_Solver` type just for this transport.
    """
    capture_handler = WarningCaptureHandler()
    solver_logger = logging.getLogger("backend.Solver")
    solver_logger.addHandler(capture_handler)
    try:
        if method_name == "Crank-Nicolson":
            simulation = CrankNicolson(
                potential, wavefunc, delta_t, grid_step=grid_step
            )
        elif method_name == "SSFM":
            simulation = SSFM(potential, wavefunc, delta_t, grid_step=grid_step)
        elif method_name == "Symmetric SSFM":
            simulation = SSFMSymmetric(
                potential, wavefunc, delta_t, grid_step=grid_step
            )
        else:
            raise ValueError(f"Unknown simulation method: {method_name}")
    finally:
        solver_logger.removeHandler(capture_handler)

    return simulation, capture_handler.captured_warnings
