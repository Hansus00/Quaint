#!/bin/python3
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
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
from Params import Params, WellType, SolverType

# load or use default params
p = argparse.ArgumentParser(description="Testing program for Quaint by Jaclav")
p.add_argument(
    "--config",
    type=str,
    required=False,
    help="path to config file",
)
p.add_argument("--f", type=str, required=False, help="show all plots at live")
p.add_argument(
    "--name", type=str, required=False, help="output directory for simulation"
)
p.add_argument(
    "--params",
    type=bool,
    required=False,
    help="Show structure and default values of simulation configuration",
)
args = p.parse_args()
if args.params != None:
    print("Default params:\n", Params())
    exit(0)
insideInteractive = args.f != None

# create directory for simulation data
now = datetime.now()
directory = ("pic/" if args.name == None else args.name + "/") + str(now) + "/"
assert not Path(directory).exists(), "Cannot override directory!"
Path(directory).mkdir(parents=True, exist_ok=False)

params = Params()
if args.config != None:
    params.read(args.config)
params.write(directory + "params.json")

# set potential
# TODO: maybe make separate Potential instances fot this?
well: Potential
if params.well_type == WellType.INFINITE_WELL:
    well = InfiniteWellPotential(params.size_x, params.size_y, params.well_height)
elif params.well_type == Params.WellType.W_SHAPED:
    well = InfiniteWellPotential(params.size_x, params.size_y, params.well_height)
    ws = WShaped(params.size_x // 4, params.size_y // 4, 3, params.well_height)
    ws_inside_grid = EmbeddedPotential(
        params.size_x,
        params.size_y,
        (params.size_x - params.size_x // 4) // 2,
        (params.size_y - params.size_y // 4) // 2,
        ws,
    )
    well += ws_inside_grid
elif params.well_type == WellType.MATRYOSHKA:
    well = InfiniteWellPotential(params.size_x, params.size_y, params.well_height)
    inside_size = (32, 32)
    inside_well = InfiniteWellPotential(
        inside_size[0], inside_size[1], params.well_height
    )
    inside_well_resized = EmbeddedPotential(
        params.size_x,
        params.size_y,
        (params.size_x - inside_size[0]) // 2,
        (params.size_y - inside_size[1]) // 2,
        inside_well,
    )
    well += inside_well_resized
elif params.well_type == WellType.NONE:
    well = InfiniteWellPotential(params.size_x, params.size_y, 0)
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
    params.r0,
    params.k0,
    params.sigma0,
    params.mass,
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
# run test
solver: _Solver
if params.solver == SolverType.CN:
    solver = CrankNicolson(well, gauss, params.delta_t)
elif params.solver == SolverType.SSFM:
    solver = SSFM(well, gauss, params.delta_t)
else:
    assert False, "Solver must be specified!"

Energies = []
Probabilities = []
start = time.perf_counter()
for i in range(0, params.steps_max):
    solver.update(params.delta_n)
    if params.solver == SolverType.CN:  # TODO: add .energy() to ssfm
        Energies.append(solver.energy())
    else:
        Energies.append(0)
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
end = time.perf_counter()

# %%
# save output
import json

with open(directory + "out.json", "w") as f:
    out = {
        "TimeOfExecution": (end - start),
        "TimeOfExecutionPerStep": (end - start) / solver.get_steps_evolved(),
        "Energies": [[complex(z).real, complex(z).imag] for z in np.array(Energies)],
        "Probabilities": [
            [complex(z).real, complex(z).imag] for z in np.array(Probabilities)
        ],
    }
    json.dump(out, f, indent=4)
# %%
# Plot P(t) and E(t)
N = [i * params.delta_n * params.delta_t for i, e in enumerate(Energies)]

fig, ax1 = plt.subplots()

ax1.plot(N, np.array(Energies) / Energies[0] - 1, label=r"$E(t)$", color="tab:blue")
ax1.set_xlabel("t")
ax1.set_ylabel("$E(t)/E(0) - 1$", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(N, np.array(Probabilities) - 1, label=r"$P(t)$", color="tab:red")
ax2.set_ylabel("$P(t)/P(0) - 1$", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
plt.savefig(directory + "EPplot.png")

if insideInteractive:
    plt.show()
plt.close()

# %%
