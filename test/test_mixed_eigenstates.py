import pytest
import numpy as np
import itertools as it

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import InfiniteWellSolver, _AnalyticSolver
from backend.Solver import CrankNicolson, SSFM, SSFMSymmetric
from backend.Potential import InfiniteWellPotential
from backend.Params import Params

size = 30
grid_step = 0.7
L = size * grid_step

delta_t = 0.0001

params = Params(L, L, grid_step, delta_t=delta_t)

well = InfiniteWellPotential(size, size)

N = 1000

mass = 2e-3


class InfiniteWellMixedSolver(_AnalyticSolver):
    def __init__(
        self,
        Nx1: int,
        Ny1: int,
        Nx2: int,
        Ny2: int,
        Lx: float,
        Ly: float,
        grid_size: float,
        mass: float,
        delta_t: float = 0.001,
        grid_step: float = 1,
    ):

        super().__init__(
            InfiniteWellPotential, None, grid_size, mass, delta_t, grid_step
        )

        # wf is only inside the well
        well_mask = (
            (0 <= self._grid[0])
            * (self._grid[0] <= Lx)
            * (0 <= self._grid[1])
            * (self._grid[1] <= Ly)
        )

        # source: https://en.wikipedia.org/wiki/Particle_in_a_box#Higher-dimensional_boxes
        energy1 = np.pi**2 / (2 * mass) * (Nx1**2 / Lx**2 + Ny1**2 / Ly**2)
        energy2 = np.pi**2 / (2 * mass) * (Nx2**2 / Lx**2 + Ny2**2 / Ly**2)

        wave_func = lambda t: well_mask * (
            2
            / np.sqrt(Lx * Ly)
            * np.sin(Nx1 * np.pi / Lx * self._grid[0])
            * np.sin(Ny1 * np.pi / Ly * self._grid[1])
            * np.exp(1j * energy1 * t)
            + 2
            / np.sqrt(Lx * Ly)
            * np.sin(Nx2 * np.pi / Lx * self._grid[0])
            * np.sin(Ny2 * np.pi / Ly * self._grid[1])
            * np.exp(1j * energy2 * t)
            / np.sqrt(2)
        )
        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )


@pytest.mark.parametrize(
    "Nx1,Ny1, Nx2, Ny2,solv_init",
    list(
        it.product(
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [CrankNicolson, SSFM, SSFMSymmetric],
        )
    ),
)
def test_stationary_evolution(Nx1, Ny1, Nx2, Ny2, solv_init):
    B = InfiniteWellMixedSolver(
        Nx1, Ny1, Nx2, Ny2, L, L, size, mass, delta_t, grid_step
    )

    solver = solv_init(well, B.get_wave_function(), params)

    for n in range(N):
        print(n)
        assert np.allclose(B.update().matrix, solver.update().matrix, )
