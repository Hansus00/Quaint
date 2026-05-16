import sys
from pathlib import Path

# to call backend module from frontend directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

# calling backend module
from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson
import numpy as np

# tests
ipw = InfiniteWellPotential(10, 10, 1e5)
gauss = GaussianPacket(
    (10, 10), np.array([1, 1]), np.array([[1, 0], [0, 1]]), 1, *ipw.matrix.shape
)
cn = CrankNicolson(ipw, gauss)
print(cn.L_2D)
print("\n\n\n")

print(cn.get_wave_function().matrix)
print("\n\n\n")
cn.step()
print(cn.get_wave_function().matrix)
print("\n\n\n")

wf = cn.update(2)
print(wf.matrix)
