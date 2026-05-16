import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson
import numpy as np

ipw = InfiniteWellPotential(10, 10, 1e5)
cn = CrankNicolson(ipw)
# print(cn.L_2D)
wf = GaussianPacket(
    (10, 10), np.array([1, 1]), np.array([[1, 0], [0, 1]]), 1, *ipw.matrix.shape
)
wf = cn(wf)
print(wf.matrix)
