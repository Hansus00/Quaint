# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import (
    Potential,
    InfiniteWellPotential,
    WShaped,
    EmbeddedPotential,
)
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson, _Solver, SSFM
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# load or use default params
params = {
    "size_x": 128,
    "size_y": 128,
    "well-type": "w-shaped",
    "well_height": 1e6,
    "solver": "cn",
    "r0": (64, 64),
    "k0": np.array([1, 0]).tolist(),
    "sigma0": np.array([[16, 0], [0, 16]]).tolist(),
    "mass": 2e-3,
    "delta_n": 32,
    "delta_t": 1e-3,
    "steps_max": 4,
}

p = argparse.ArgumentParser()
p.add_argument("--config", type=str, required=False)
p.add_argument("--f", type=str, required=False)
p.add_argument("--name", type=str, required=False)
args = p.parse_args()
insideInteractive = args.f != None

# create directory for simulation data
now = datetime.now() if args.name == None else args.name
directory = "pic/" + str(now) + "/"
assert not Path(directory).exists(), "Cannot override directory!"
Path(directory).mkdir(parents=True, exist_ok=False)

if args.config != None:
    with open(args.config, "r") as f:
        params = json.load(f)
with open(directory + "params.json", "w") as f:
    json.dump(params, f, indent=4)

# set potential
well: Potential
if params["well-type"] == "infiniteWell":  # TODO: use enum instead of string
    well = InfiniteWellPotential(
        params["size_x"], params["size_y"], params["well_height"]
    )
elif params["well-type"] == "w-shaped":
    well = InfiniteWellPotential(
        params["size_x"], params["size_y"], params["well_height"]
    )
    ws = WShaped(params["size_x"] // 4, params["size_y"] // 4, 3, params["well_height"])
    ws_inside_grid = EmbeddedPotential(
        params["size_x"],
        params["size_y"],
        (params["size_x"] - params["size_x"] // 4) // 2,
        (params["size_y"] - params["size_y"] // 4) // 2,
        ws,
    )
    well += ws_inside_grid
elif params["well-type"] == "matryoshka":
    well = InfiniteWellPotential(
        params["size_x"], params["size_y"], params["well_height"]
    )
    inside_size = (32, 32)
    inside_well = InfiniteWellPotential(
        inside_size[0], inside_size[1], params["well_height"]
    )
    inside_well_resized = EmbeddedPotential(
        params["size_x"],
        params["size_y"],
        (params["size_x"] - inside_size[0]) // 2,
        (params["size_y"] - inside_size[1]) // 2,
        inside_well,
    )
    well += inside_well_resized
elif params["well-type"] == "none":
    well = InfiniteWellPotential(params["size_x"], params["size_y"], 0)
else:
    assert False, "Potential must be specified!"

# draw potential
plt.style.use("JK_W.mplstyle")
plt.title("Potential well")
im = plt.imshow(well.matrix)
cbar = plt.colorbar(im)
cbar.set_label(r"$V(x,y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(directory + "gauss_evolved_n0000.png")
if insideInteractive:
    plt.show()
plt.close()

# set and draw psi(0)
gauss = GaussianPacket(
    params["r0"],
    params["k0"],
    params["sigma0"],
    params["mass"],
    *well.matrix.shape,
)  # TODO: Test Airy wave train #33
plt.title("Gaussian packet at start")
im = plt.imshow(np.abs(gauss.matrix) ** 2)
cbar = plt.colorbar(
    im, format="%.4f"
)  # FIXME: make it look good, maybe scientific notation?
cbar.set_label(r"$|\psi|^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(directory + "gauss_evolved_n0001.png")
if insideInteractive:
    plt.show()
plt.close()

# %%
delta_n = params["delta_n"]
solver: _Solver
if params["solver"] == "cn":
    solver = CrankNicolson(well, gauss, params["delta_t"])
elif params["solver"] == "ssfm":
    solver = SSFM(well, gauss, params["delta_t"])
else:
    assert False, "Solver must be specified!"

Energies = []
Probabilities = []
for i in range(0, params["steps_max"]):
    solver.update(delta_n)
    if params["solver"] == "cn":  # TODO: add .energy() to ssfm
        Energies.append(solver.energy())
    Probabilities.append(solver.get_wave_function().total_probability())

    plt.title("Evolved gaussian packet n=" + str(solver.get_steps_evolved()))
    im = plt.imshow(np.abs(solver.get_wave_function().matrix) ** 2)
    cbar = plt.colorbar(
        im, format="%.4f"
    )  # FIXME: make it look good, maybe scientific notation?
    cbar.set_label(r"$|\psi|^2$")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(
        directory + "gauss_evolved_n" + f"{solver.get_steps_evolved():04d}" + ".png"
    )
    if insideInteractive:
        plt.show()
    plt.close()


# %%
# save output
with open(directory + "Energies.out.json", "w") as f:
    out = np.array(Energies)
    json.dump([[complex(z).real, complex(z).imag] for z in out], f, indent=4)
with open(directory + "Probabilities.out.json", "w") as f:
    out = np.array(Probabilities)
    json.dump([[complex(z).real, complex(z).imag] for z in out], f, indent=4)
# %%
N = [i * delta_n * params["delta_t"] for i, e in enumerate(Energies)]

fig, ax1 = plt.subplots()

# oś Y: energie
ax1.plot(N, np.array(Energies) / Energies[0] - 1, label=r"$E(t)$", color="tab:blue")
ax1.set_xlabel("t")
ax1.set_ylabel("$E(t)/E(0) - 1$", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# druga oś Y: prawdopodobieństwo
ax2 = ax1.twinx()
ax2.plot(N, np.array(Probabilities) - 1, label=r"$P(t)$", color="tab:red")
ax2.set_ylabel("$P(t)/P(0) - 1$", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
plt.savefig(directory + "EPplot.png")

if insideInteractive:
    plt.show()
plt.close()

# %%
