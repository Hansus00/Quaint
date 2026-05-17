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
elif params.well_type == WellType.W_SHAPED:
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
    inside_size = (params.size_x // 3, params.size_y // 3)
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
im = plt.imshow(np.float64(well.matrix))
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
im = plt.imshow(np.float64(np.abs(gauss.matrix) ** 2))
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
        Energies.append(1)
    Probabilities.append(solver.get_wave_function().total_probability())

    plt.title("Evolved gaussian packet n=" + str(solver.get_steps_evolved()))
    im = plt.imshow(np.float64(np.abs(solver.get_wave_function().matrix) ** 2))
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
        "time_of_execution": (end - start),
        "time_of_execution_per_step": (end - start) / solver.get_steps_evolved(),
        "energy_abs_stdev": [np.std(np.abs(Energies))],
        "energies": [[complex(z).real, complex(z).imag] for z in np.array(Energies)],
        "probabilities": np.array(np.abs(Probabilities), dtype=float).tolist(),
    }
    json.dump(out, f, indent=4)
# %%
# Plot P(t) and E(t)
N = [i * params.delta_n * params.delta_t for i, e in enumerate(Energies)]

fig, ax1 = plt.subplots()

ax1.plot(
    N,
    np.array(Energies).real / np.abs(Energies[0]) - 1,
    label=r"$\Re \left(E(t)/|E(0)|-1\right)$",
    color="tab:blue",
)

assert np.allclose(
    np.imag(Energies), np.zeros_like(Energies)
), "\nERROR: Expected values of hermitian operators are real"

ax1.plot(
    N,
    np.array(Energies).imag / np.abs(Energies[0]),
    label=r"$\Im E(t)/|E(0)|$",
    linestyle="--",
    color="tab:blue",
)
ax1.set_xlabel("t")
ax1.set_ylabel("Energy $E$ (arb. u.)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(N, np.array(Probabilities) - 1, label=r"$P(t)$", color="tab:red")
ax2.set_ylabel("Probability change $P(t)- 1$", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

leg = ax1.legend(frameon=True, framealpha=1)
leg.set_zorder(100)

plt.savefig(directory + "EPplot.png")

if insideInteractive:
    plt.show()
plt.close()

# %%
