#!/bin/python3
# %%
import sys
from pathlib import Path
from importlib.resources import files

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
)
from backend.Potential import (
    Potential,
    InfiniteWellPotential,
    Slab,
    WShaped,
    EmbeddedPotential,
)
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson, _Solver, SSFM, SSFMSymmetric
from backend.Params import Params, PotentialType, SolverType
from backend.Analytic import GaussianPacketSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
import time
import argparse
import json
import matplotlib.animation as animation
import colorsys as cs
import logging
import frontend.LoggerTools as LoggerTools

logger = logging.getLogger(__name__)
LoggerTools.configLogger(LoggerTools.INFO)


# load or use default params
p = argparse.ArgumentParser(description="Testing program for Quaint by Jaclav")
p.add_argument(
    "--config",
    type=str,
    required=False,
    metavar="CONFIG_FILE",
    help="Path to the configuration file",
)
p.add_argument(
    "--f",
    type=str,
    required=False,
    help="Show plots live, will not save output .mp4, is faster",
)  # required by interactive mode
p.add_argument(
    "-f", action="store_true", help="Same as --f"
)  # required by interactive mode
p.add_argument(
    "--out",
    type=str,
    required=False,
    help="Set output directory for simulation",
    metavar="OUTPUT_PATH",
)
p.add_argument(
    "--fps",
    type=float,
    default=5,
    required=False,
    help="Set FPS rate for animation",
    metavar="FPS",
)
p.add_argument(
    "--solver",
    choices=[e.value for e in SolverType],
    required=False,
    help="Set solving algorithm (overrides --config)",
)
p.add_argument(
    "--T-tot",
    type=int,
    required=False,
    help="Set how many updates, each of delta_n, should happen (overrides --config)",
)
p.add_argument(
    "--grid-step",
    type=float,
    required=False,
    help="Set size of grid step",
)
p.add_argument(
    "--params",
    action="store_true",
    required=False,
    help="Show structure and default values of simulation configuration",
)
p.add_argument(
    "--do-not-animate",
    action="store_true",
    required=False,
    help="Will produce no animation, as it requires ffmpeg",
)
args = p.parse_args()
if args.params:
    logger.info("Default params:\n" + str(Params()))
    exit(0)

insideInteractive = args.f is not None
if args.do_not_animate:
    insideInteractive = False

# create directory for simulation data
now = str(datetime.now()).replace(":", "-").replace(" ", "_")
base_dir = Path("pic") if args.out is None else Path(args.out)
directory = base_dir / str(now)
assert not directory.exists(), "Cannot override directory!"
directory.mkdir(parents=True, exist_ok=False)

params = Params()
if args.config is not None:
    params.read(args.config)
if args.solver is not None:
    params.solver = args.solver
if args.T_tot is not None:
    params.T_tot = args.T_tot
if args.grid_step is not None:
    params.grid_step = args.grid_step
params.write(str(directory / "params.json"))
logger.info("Simulation parameters: %s", params)

# set potential
# TODO: maybe make separate Potential instances for this?
well = InfiniteWellPotential(params.grid_size_x, params.grid_size_y)
if params.potential_type == PotentialType.INFINITE_WELL:
    pass
elif params.potential_type == PotentialType.W_SHAPED:
    ws = WShaped(
        params.grid_size_x // 4, params.grid_size_y // 4, 3, params.well_height
    )
    ws_inside_grid = EmbeddedPotential(
        params.grid_size_x,
        params.grid_size_y,
        (params.grid_size_x - params.grid_size_x // 4) // 6,  # check for asymmetry
        (params.grid_size_y - params.grid_size_y // 4) // 2,
        ws,
    )
    well += ws_inside_grid
elif params.potential_type == PotentialType.MATRYOSHKA:
    inside_size = (params.grid_size_x // 3, params.grid_size_y // 3)
    inside_well = InfiniteWellPotential(inside_size[0], inside_size[1])
    inside_well_resized = EmbeddedPotential(
        params.grid_size_x,
        params.grid_size_y,
        (params.grid_size_x - inside_size[0]) // 2,
        (params.grid_size_y - inside_size[1]) // 2,
        inside_well,
    )
    well += inside_well_resized
elif params.potential_type == PotentialType.SLAB:
    slab = Slab(params.grid_size_x // 16, params.grid_size_y, params.well_height)
    well += EmbeddedPotential(
        params.grid_size_x,
        params.grid_size_y,
        (params.grid_size_x - params.grid_size_x // 16) // 2,
        0,
        slab,
    )
elif params.potential_type == PotentialType.DOUBLE_SLIT:
    slab = Slab(1, params.grid_size_y, params.well_height)

    slit = Slab(1, 8, params.well_height)
    slab -= EmbeddedPotential(
        slab.matrix.shape[0],
        slab.matrix.shape[1],
        0,
        params.grid_size_y // 2 + 3,
        slit,
    )
    slab -= EmbeddedPotential(
        slab.matrix.shape[0],
        slab.matrix.shape[1],
        0,
        params.grid_size_y // 2 - 8 - 3,
        slit,
    )

    well += EmbeddedPotential(
        params.grid_size_x,
        params.grid_size_y,
        (params.grid_size_x - params.grid_size_x // 16) // 4,
        0,
        slab,
    )
else:
    assert False, "Potential must be specified!"

# draw potential
plt.style.use("JK_W.mplstyle")  # type: ignore
fig, ax = plt.subplots(layout="tight")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ax.set_title("Potential well")
im = ax.imshow(
    np.float64(well.matrix).T,
    aspect="auto",
    origin="lower",
    norm=LogNorm(
        vmin=np.min(np.float64(well.matrix)) + 0.001,
        vmax=np.max(np.float64(well.matrix)) + 0.01,
    ),
)  # transposition is needed as imshow draws (y,x)
cbar = plt.colorbar(im)
cbar.set_label(r"$V(x,y)$")
ax.set_xlabel("x")
ax.set_ylabel("y")

# set and draw psi(0)
gauss = GaussianPacket(
    params.r0_grid,
    params.k0_grid,
    params.sigma0_grid,
    *well.matrix.shape,
)  # TODO: Test Airy wave train #33

# %%
# run test
# %matplotlib widget
solver: _Solver
if params.solver == SolverType.CN:
    solver = CrankNicolson(well, gauss, params)
elif params.solver == SolverType.SSFM:
    solver = SSFM(well, gauss, params)
elif params.solver == SolverType.SYM_SSFM:
    solver = SSFMSymmetric(well, gauss, params)
elif params.solver == SolverType.ANALYTIC_GAUSSIAN:
    well = InfiniteWellPotential(params.grid_size_x, params.grid_size_y)
    raise NotImplementedError
else:
    assert False, "Solver must be specified!"

Energies: list[complex] = []
Probabilities = []
start = time.perf_counter()


# main simulation
FRAMES_FOR_POTENTIAL = 3


def update(frame):
    logger.info("frame no. %d step %d", frame, solver.get_steps_evolved())
    """0th frame is potential"""
    if frame < FRAMES_FOR_POTENTIAL:
        return (im,)
    elif frame == FRAMES_FOR_POTENTIAL:
        cbar.ax.set_visible(False)
    else:
        solver.update()
        Energies.append(solver.ev_energy())  # type: ignore
        Probabilities.append(solver.get_wave_function().total_probability())

    new_data = solver.get_wave_function().matrix
    ax.set_title(
        "Evolved ("
        + str(params.solver)
        + ") gaussian packet n="
        + str(solver.get_steps_evolved())
    )
    amp = np.abs(new_data) ** 2
    scale = np.percentile(amp, 99.9)

    amp = np.clip(amp / scale, 0, 1)

    colors = [
        [
            cs.hls_to_rgb(
                (np.angle(z) + np.pi) / (2 * np.pi),
                a,
                1,
            )
            for z, a in zip(row, amp_row)
        ]
        for row, amp_row in zip(
            new_data.T, amp.T
        )  # transposition is needed as imshow draws (y,x)
    ]
    im.set_data(colors)
    return (im,)


if args.do_not_animate:
    for i in range(0, int(params.T_tot / params.delta_t) + FRAMES_FOR_POTENTIAL + 2):
        update(i)
else:
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, int(params.T_tot / params.delta_t) + FRAMES_FOR_POTENTIAL + 2),
        interval=1e3 / args.fps,
        blit=False,
        repeat=False,
    )  # type: ignore

    if not insideInteractive:
        try:
            writer = animation.FFMpegWriter(
                fps=args.fps,
                codec="libx264",
                extra_args=[
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "4.0",
                    "-preset",
                    "medium",
                    "-movflags",
                    "+faststart",
                ],
            )
            ani.save(
                directory / f"gauss_evolution.mp4",
                writer=writer,
                dpi=300,
                savefig_kwargs={"pad_inches": 0},
            )  # similiar as ffmpeg -framerate 2 -pattern_type glob -i "gauss_evolved_n*.png" output.mp4
        except Exception as e:
            logger.critical("ffmpeg missing or broken:", e)
            exit(-1)
    else:
        plt.show()
end = time.perf_counter()
logger.info("time_of_execution %f", float(end - start))


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
N = [i * params.delta_t for i, e in enumerate(Energies)]

fig, ax1 = plt.subplots()
ax1.set_title(
    "Probability and expected value of energy during time evolution ("
    + str(params.solver)
    + ")"
)
ax1.plot(
    N,
    np.array(Energies).real / np.abs(Energies[0]) - 1,  # type: ignore
    label=r"$\Re \langle E\rangle(t)/\left|\langle E\rangle(0)\right|-1$",
    color="tab:blue",
)

assert np.allclose(
    np.imag(Energies), np.zeros_like(Energies)
), "\nERROR: Expected values of hermitian operators are real"

ax1.plot(
    N,
    np.array(Energies).imag / np.abs(Energies[0]),  # type: ignore
    label=r"$\Im \langle E\rangle(t)/\left|\langle E\rangle(0)\right|$",
    linestyle="--",
    color="tab:blue",
)
ax1.set_xlabel("t")
ax1.set_ylabel("Energy $E$ (arb. u.)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(N, np.array(Probabilities) - 1, label=r"$P(t)$", color="tab:red")
ax2.set_ylabel("Probability deviation $P(t)-1$", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

leg = ax1.legend(frameon=True, framealpha=1, loc="upper left")
leg.set_zorder(100)

plt.savefig(directory / "EPplot.png")

if insideInteractive:
    plt.show()
plt.close()

# %%
