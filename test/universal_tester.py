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
import json
from Params import Params, WellType, SolverType
import matplotlib.animation as animation

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
if args.params is not None:
    print("Default params:\n", Params())
    exit(0)
insideInteractive = args.f is not None

# create directory for simulation data
now = datetime.now()
base_dir = Path("pic") if args.name is None else Path(args.name)
directory = base_dir / str(now)
assert not directory.exists(), "Cannot override directory!"
directory.mkdir(parents=True, exist_ok=False)

params = Params()
if args.config is not None:
    params.read(args.config)
params.write(str(directory / "params.json"))

# set potential
# TODO: maybe make separate Potential instances for this?
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
fig, ax = plt.subplots(layout="tight")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ax.set_title("Potential well")
im = ax.imshow(np.float64(well.matrix), aspect="auto")
cbar = plt.colorbar(im)
cbar.set_label(r"$V(x,y)$")
ax.set_xlabel("x")
ax.set_ylabel("y")

# set and draw psi(0)
gauss = GaussianPacket(
    params.r0,
    params.k0,
    params.sigma0,
    params.mass,
    *well.matrix.shape,
)  # TODO: Test Airy wave train #33


# %%
# run test
# %matplotlib widget
solver: _Solver
if params.solver == SolverType.CN:
    solver = CrankNicolson(well, gauss, params.delta_t)
elif params.solver == SolverType.SSFM:
    solver = SSFM(well, gauss, params.delta_t)
else:
    assert False, "Solver must be specified!"

Energies: list[complex] = []
Probabilities = []
start = time.perf_counter()


def update(frame):
    print("frame no", frame, "n", str(solver.get_steps_evolved()))
    """0th frame is potential"""
    if frame < 1:
        return (im,)
    elif frame == 1:
        cbar.set_label(r"$|\psi|^2$")
    else:
        solver.update(params.delta_n)
        if params.solver == SolverType.CN:  # TODO: add .energy() to ssfm
            Energies.append(solver.energy())  # type: ignore
        else:
            Energies.append(1)
        Probabilities.append(solver.get_wave_function().total_probability())

    new_data = np.float64(np.abs(solver.get_wave_function().matrix) ** 2)
    im.set_clim(vmin=new_data.min(), vmax=new_data.max())
    cbar.update_normal(im)
    ax.set_title("Evolved gaussian packet n=" + str(solver.get_steps_evolved()))
    im.set_data(new_data)
    return (im,)


ani = animation.FuncAnimation(
    fig, update, frames=range(0, params.updates_max + 3), interval=200, blit=False
)  # type: ignore

end = time.perf_counter()
ani.save(
    directory / f"gauss_evolution.mp4",
    writer="ffmpeg",
    fps=10,
    dpi=300,
    savefig_kwargs={"pad_inches": 0},
)

# %%
# save output
with open(directory / "out.json", "w") as f:
    out = {
        "time_of_execution": float(end - start),
        "time_of_execution_per_step": float((end - start) / solver.get_steps_evolved()),
        "energy__exp_val_abs_stdev": float(np.std(np.abs(Energies))),
        "energy_exp_val": [[z.real, z.imag] for z in np.array(Energies).tolist()],
        "probabilities": np.array(np.abs(Probabilities), dtype=float).tolist(),
    }
    json.dump(out, f, indent=4)
# %%
# Plot P(t) and E(t)
N = [i * params.delta_n * params.delta_t for i, e in enumerate(Energies)]

fig, ax1 = plt.subplots()

ax1.plot(
    N,
    np.array(Energies).real / np.abs(Energies[0]) - 1,  # type: ignore
    label=r"$\Re \left(E(t)/|E(0)|-1\right)$",
    color="tab:blue",
)

assert np.allclose(
    np.imag(Energies), np.zeros_like(Energies)
), "\nERROR: Expected values of hermitian operators are real"

ax1.plot(
    N,
    np.array(Energies).imag / np.abs(Energies[0]),  # type: ignore
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

plt.savefig(directory / "EPplot.png")

if insideInteractive:
    plt.show()
plt.close()

# %%
