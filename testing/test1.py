# %%
import sys
from pathlib import Path

# to call backend module from frontend directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson, Constant
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("JK_W.mplstyle")
ipw = InfiniteWellPotential(128, 128, 1e6)  # Mock potential well
plt.imshow(ipw.matrix)
plt.show()

gauss = GaussianPacket(
    (64, 64),
    np.array([0, 0]),
    np.array([[16, 0], [0, 16]]),
    0.001,
    *ipw.matrix.shape,
)
plt.title("Gaussian packet at start")
plt.imshow(np.abs(gauss.matrix))
plt.savefig("gauss_start.png")
plt.show()

cn = CrankNicolson(ipw, gauss)
n_steps = 64
cn.update(n_steps)
plt.title("Evolved gaussian packet n=" + str(n_steps))
plt.imshow(np.abs(cn.get_wave_function().matrix))
plt.savefig("gauss_evolved.png")
plt.show()


# %%
