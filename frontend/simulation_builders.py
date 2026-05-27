import logging
from typing import Any

import numpy as np
from backend.Potential import Potential
from backend.Solver import SSFM, Constant, CrankNicolson
from backend.StationaryWaveFunc import GaussianPacket

from .warning_handler import WarningCaptureHandler


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
) -> Any:
    """Instantiate selected solver and attach captured stability warnings."""
    capture_handler = WarningCaptureHandler()
    solver_logger = logging.getLogger("backend.Solver")
    solver_logger.addHandler(capture_handler)
    try:
        if method_name == "Constant":
            simulation = Constant(potential, wavefunc, delta_t, grid_step=grid_step)
        elif method_name == "Crank-Nicolson":
            simulation = CrankNicolson(
                potential, wavefunc, delta_t, grid_step=grid_step
            )
        elif method_name == "SSFM":
            simulation = SSFM(potential, wavefunc, delta_t, grid_step=grid_step)
        else:
            raise ValueError(f"Unknown simulation method: {method_name}")
    finally:
        solver_logger.removeHandler(capture_handler)

    # TODO: fix typing (_Solver has no field stability_warnings)
    setattr(simulation, "stability_warnings", capture_handler.captured_warnings)
    return simulation
