# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson, Constant
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
directory = "pic/" + str(now) + "/"
Path(directory).mkdir(parents=True, exist_ok=False)

params = {
    "size_x": 128,
    "size_y": 128,
    "well_height": 1e6,
    "r0": (64, 64),
    "k0": np.array([1, 0]).tolist(),
    "sigma0": np.array([[16, 0], [0, 16]]).tolist(),
    "mass": 1e-3,
    "delta_n": 32,
    "updates_max": 128,
}
with open(directory + "dump.json", "w") as f:
    json.dump(params, f, indent=4)

plt.style.use("JK_W.mplstyle")
ipw = InfiniteWellPotential(params["size_x"], params["size_y"], params["well_height"])
plt.title("Potential well")
plt.imshow(ipw.matrix)
plt.savefig(directory + "gauss_evolved_0000.png")
plt.show()

gauss = GaussianPacket(
    params["r0"],
    params["k0"],
    params["sigma0"],
    params["mass"],
    *ipw.matrix.shape,
)
plt.title("Gaussian packet at start")
plt.imshow(np.abs(gauss.matrix))
plt.savefig(directory + "gauss_evolved_0001.png")
plt.show()

# %%
cn = CrankNicolson(ipw, gauss)
delta_n = params["delta_n"]

for i in range(0, params["steps_max"]):
    cn.update(delta_n)
    plt.title("Evolved gaussian packet n=" + str(cn.get_steps_evolved()))
    plt.imshow(np.abs(cn.get_wave_function().matrix))
    plt.savefig(
        directory + "gauss_evolved_n" + f"{cn.get_steps_evolved():04d}" + ".png"
    )
    plt.show()


# %%
