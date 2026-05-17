# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential, WShaped, EmbeddedPotential
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
    "well-type": "w-shaped",
    "well_height": 1e6,
    "r0": (32, 64),
    "k0": np.array([1, 0]).tolist(),
    "sigma0": np.array([[16, 0], [0, 16]]).tolist(),
    "mass": 2e-3,
    "delta_n": 32,
    "steps_max": 16,
}
with open(directory + "dump.json", "w") as f:
    json.dump(params, f, indent=4)

well = InfiniteWellPotential(params["size_x"], params["size_y"], params["well_height"])
if params["well-type"] == "infiniteWell":
    pass
elif params["well-type"] == "w-shaped":
    ws = WShaped(params["size_x"] // 4, params["size_y"] // 4, 3, params["well_height"])
    ws_inside_grid = EmbeddedPotential(
        params["size_x"],
        params["size_y"],
        (params["size_x"] - params["size_x"] // 4) / 2,
        (params["size_y"] - params["size_y"] // 4) / 2,
        ws,
    )
    well += ws_inside_grid


plt.style.use("JK_W.mplstyle")
plt.title("Potential well")
im = plt.imshow(well.matrix)
cbar = plt.colorbar(im)
cbar.set_label(r"$V(x,y)$")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig(directory + "gauss_evolved_n0000.png")
plt.show()

gauss = GaussianPacket(
    params["r0"],
    params["k0"],
    params["sigma0"],
    params["mass"],
    *well.matrix.shape,
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

plt.savefig(directory + "gauss_evolved_n0001.png")
plt.show()

# %%
cn = CrankNicolson(well, gauss)
delta_n = params["delta_n"]

for i in range(0, params["steps_max"]):
    cn.update(delta_n)
    plt.title(
        "Evolved gaussian packet n="
        + str(cn.get_steps_evolved())
        + ",\t"
        + r"$\sum_i\,|\psi_i|^2=$"
        + "%.4f" % (cn.get_wave_function().total_probability())
    )
    im = plt.imshow(np.abs(cn.get_wave_function().matrix) ** 2)
    cbar = plt.colorbar(
        im, format="%.4f"
    )  # FIXME: make it look good, maybe scientific notation?
    cbar.set_label(r"$|\psi|^2$")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(
        directory + "gauss_evolved_n" + f"{cn.get_steps_evolved():04d}" + ".png"
    )
    plt.show()


# %%
