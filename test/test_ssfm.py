# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import SSFM
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
    "mass": 2e-3,
    "delta_n": 32,
    "steps_max": 128,
}
with open(directory + "dump.json", "w") as f:
    json.dump(params, f, indent=4)

plt.style.use("JK_W.mplstyle")
ipw = InfiniteWellPotential(params["size_x"], params["size_y"], params["well_height"])
plt.title("Potential well")
im = plt.imshow(ipw.matrix)
cbar = plt.colorbar(im)
cbar.set_label(r"$V(x,y)$")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig(directory + "gauss_evolved_0000.png")
plt.show()

gauss = GaussianPacket(
    params["r0"],
    params["k0"],
    params["sigma0"],
    params["mass"],
    *ipw.matrix.shape,
)
plt.title(
    "Gaussian packet at start\t"
    + r"$\sum_i\,|\psi_i|^2=$"
    + "%.4f" % (gauss.total_probability())
)
im = plt.imshow(np.abs(gauss.matrix) ** 2)
cbar = plt.colorbar(
    im, format="%.4f"
)  # FIXME: make it look good, maybe scientific notation?
cbar.set_label(r"$|\psi|^2$")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig(directory + "gauss_evolved_0001.png")
plt.show()

# %%
ssfm = SSFM(ipw, gauss)
delta_n = params["delta_n"]

for i in range(0, params["steps_max"]):
    ssfm.update(delta_n)
    plt.title(
        "Evolved gaussian packet SSFM n="
        + str(ssfm.get_steps_evolved())
        + ",\t"
        + r"$\sum_i\,|\psi_i|^2=$"
        + "%.4f" % (ssfm.get_wave_function().total_probability())
    )
    im = plt.imshow(np.abs(ssfm.get_wave_function().matrix) ** 2)
    cbar = plt.colorbar(
        im, format="%.4f"
    )  # FIXME: make it look good, maybe scientific notation?
    cbar.set_label(r"$|\psi|^2$")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(
        directory + "gauss_evolved_n" + f"{ssfm.get_steps_evolved():04d}" + ".png"
    )
    plt.show()


# %%
