import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential, HarmonicPotential, GaussianBumpPotential
from backend.StationaryWaveFunc import GaussianPacket
from TestUtil import CrankNicolsonTest, test_energy_conservation
import numpy as np
import matplotlib.pyplot as plt

Nx = 20
Ny = 20

N = 2000
eps = 1e-5

wf = GaussianPacket([10, 10], np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1, Nx, Ny)

# infinite well
V = InfiniteWellPotential(Nx, Ny)
nc = CrankNicolsonTest(V, wf)

test_energy_conservation(nc, N, eps)

# gaussian bump
V = GaussianBumpPotential(Nx, Ny, (0,0), 2, np.array([[1,0],[0,1]]))
nc = CrankNicolsonTest(V, wf)

test_energy_conservation(nc, N, eps)

print('Test passed')