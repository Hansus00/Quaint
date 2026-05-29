import pytest
import numpy as np
import itertools as it

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import InfiniteWellSolver
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

Nxmin = 1
Nxmax = 6
Nymin = 1
Nymax = 6

@pytest.mark.parametrize(
    "Nx,Ny,solv_init",
    list(
        it.product(
            range(Nxmin, Nxmax),
            range(Nymin, Nymax),
            [CrankNicolson, SSFM, SSFMSymmetric],
        )
    ),
)
def test_stationary_evolution(Nx, Ny, solv_init):
    B = InfiniteWellSolver(Nx, Ny, L, L, size, mass, delta_t, grid_step)

    solver = solv_init(well, B.get_wave_function(), params)

    for n in range(N):
        assert np.allclose(B.update().matrix, solver.update().matrix, atol=1e-1)
