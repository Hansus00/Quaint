import numpy as np

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import InfiniteWellBasisSolver
from backend.StationaryWaveFunc import GaussianPacket

wf = GaussianPacket(
    (50, 50),
    np.array([0, 0]),
    sigma0=[[5, 0], [0, 5]],
    size_x=100,
    size_y=100,
)

N = 150
Nx, Ny = N, N

bs = InfiniteWellBasisSolver(wf, Nx, Ny)


def test_wf_approximation():
    assert np.allclose(bs.get_wave_function().matrix, wf.matrix, atol=1e-2)
