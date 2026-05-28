import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from backend.Params import Params, SolverType
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


def build_params(
    method_name: str,
    x_limit: float,
    y_limit: float,
    grid_step: float,
    r0: np.ndarray | tuple[float, float],
    k0: np.ndarray,
    sigma0: np.ndarray,
    mass: float,
    delta_t: float,
    well_height: float,
) -> Params:
    """Standardized builder for backend Params from frontend UI primitives."""
    method_map = {
        "Crank-Nicolson": SolverType.CN,
        "SSFM": SolverType.SSFM,
        "Symmetric SSFM": SolverType.SYM_SSFM,
    }
    solver_type = method_map.get(method_name, SolverType.SSFM)

    return Params(
        length_x=x_limit,
        length_y=y_limit,
        grid_step=grid_step,
        solver=solver_type,
        r0=tuple(np.asarray(r0) * grid_step),
        k0=k0,
        sigma0=np.asarray(sigma0) * (grid_step**2),
        mass=mass,
        delta_t=delta_t,
        well_height=well_height,
    )


def instantiate_solver_with_warnings(
    potential: Potential,
    wavefunc: GaussianPacket,
    params: Params
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
        if params.solver == SolverType.CN:
            simulation = CrankNicolson(potential, wavefunc, params)
        elif params.solver == SolverType.SSFM:
            simulation = SSFM(potential, wavefunc, params)
        elif params.solver == SolverType.SYM_SSFM:
            simulation = SSFMSymmetric(potential, wavefunc, params)
        else:
            raise ValueError(f"Unknown simulation method: {params.solver}")
    finally:
        solver_logger.removeHandler(capture_handler)

    return simulation, capture_handler.captured_warnings
