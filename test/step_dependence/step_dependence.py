import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from itertools import product
import pickle
from typing import Any
from copy import deepcopy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.Solver import CrankNicolson, SSFMSymmetric, SSFM
from backend.Params import Params
from backend.Analytic import _AnalyticSolver
from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import StationaryWaveFunc


def wave_function(
    modes: list[list[int, int]] | NDArray[np.int32],
    coeffs: list[complex] | NDArray[np.complex128],
    sizex: int,
    sizey: int,
    grid_step: float,
) -> StationaryWaveFunc:
    """Creates the wave function as a normalized weighted superposition
    of the basis functions inside the 2D infinite well.
    If N[i][0] = Nx and N[i][1] = Ny then coeff[i] is the coefficient
    of sin(pi * Nx * x / Lx) * sin(pi * Ny * y / Ly)."""

    Lx = sizex * grid_step
    Ly = sizey * grid_step

    pos1dx = np.linspace(0, Lx, sizex)
    pos1dy = np.linspace(0, Ly, sizey)

    modes = np.array(modes)
    coeffs = np.array(coeffs)

    # normalization
    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    coeffs = coeffs / norm

    wave_matrix = np.sum(  # sum over modes
        [
            coeff
            * np.outer(
                np.sqrt(2 / Lx) * np.sin(Nx * np.pi / Lx * pos1dx),
                np.sqrt(2 / Ly) * np.sin(Ny * np.pi / Ly * pos1dy),
            )  # basis function for mode (Nx, Ny)
            for coeff, (Nx, Ny) in zip(coeffs, modes)
        ],
        axis=0,
    )

    return StationaryWaveFunc(wave_matrix)


class InfiniteWellMixedSolver(_AnalyticSolver):
    def __init__(
        self,
        modes: list[list[int, int]] | NDArray[np.int32],
        coeffs: list[complex] | NDArray[np.complex128],
        mass: float,
        sizex: int,
        sizey: int,
        grid_step: float,
        delta_t: float = 0.001,
    ):
        """Creates analytic solver which calculates the evolution
        of wavefunctions consisting of a finite linear combination
        of infinite well hamiltonian eigenstates."""

        self.delta_t = delta_t
        self.mass = mass

        Lx = sizex * grid_step
        Ly = sizey * grid_step

        self._pos1dx = np.linspace(0, Lx, sizex)
        self._pos1dy = np.linspace(0, Ly, sizey)

        self._modes = np.array(modes)
        coeffs = np.array(coeffs)

        # normalization
        norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
        coeffs = coeffs / norm

        # eigenstate basis
        self._basis = [
            np.outer(
                np.sqrt(2 / Lx) * np.sin(Nx * np.pi / Lx * self._pos1dx),
                np.sqrt(2 / Ly) * np.sin(Ny * np.pi / Ly * self._pos1dy),
            )
            for (Nx, Ny) in self._modes
        ]

        # source: https://en.wikipedia.org/wiki/Particle_in_a_box#Higher-dimensional_boxes
        self._energies = (
            np.pi**2
            / (2 * mass)
            * ((self._modes[:, 0] / Lx) ** 2 + (self._modes[:, 1] / Ly) ** 2)
        )

        self._wave_lambda = lambda t: (
            np.sum(  # sum over modes
                [
                    coeff
                    * np.exp(-1j * energy * t)  # including standard evolution
                    * basis_vector
                    for coeff, energy, basis_vector in zip(
                        coeffs, self._energies, self._basis
                    )
                ],
                axis=0,
            )
        )


def l2(A: NDArray[np.complex128], dx: float, dy: float) -> float:
    """Calculates L2 norm of A by approximating the integral as a simple sum assuming
    a regular rectangular grid"""
    return np.sum(np.abs(A) ** 2) * dx * dy


def sup(A: NDArray[np.complex128]) -> float:
    """Calculates sup norm as the maximum of absolute value of A."""
    return np.max(np.abs(A))


class TimeStepper:
    """Container class for the logic of creating solvers for each
    time step value and comparing the outputs to exact solution."""

    def __init__(self):
        pass

    def calc_errors(
        self,
        dt_space: NDArray[np.float64],
        params: Params,
        modes: list[list[int, int]] | NDArray[np.int32],
        coeffs: list[complex] | NDArray[np.complex128],
    ):
        """Calculates the differences between each solver and the exact
        solution after one step for an array of time step sizes."""

        self.dt_space = dt_space
        self.params = params
        self.modes = modes
        self.coeffs = coeffs

        initial_wave_function = wave_function(
            modes, coeffs, params.grid_size_x, params.grid_size_y, params.grid_step
        )
        potential = InfiniteWellPotential(params.grid_size_x, params.grid_size_y)

        solver_names = ["cn", "ssfm", "sym_ssfm"]

        norm_names = ["sup", "l2"]
        norms = [
            lambda a, b: sup(a.matrix - b.matrix),
            lambda a, b: l2(a.matrix - b.matrix, params.dx, params.dy),
        ]

        self.results = {
            (solvername, normname): []
            for (solvername, normname) in product(solver_names, norm_names)
        }

        # calculate errors for each time step size
        for dt in dt_space:
            updated_params = deepcopy(params)
            updated_params.delta_t = dt

            # create solvers
            exact_solver = InfiniteWellMixedSolver(
                modes,
                coeffs,
                params.mass,
                params.grid_size_x,
                params.grid_size_y,
                params.grid_step,
                dt,
            )
            crank_nicolson = CrankNicolson(
                potential, initial_wave_function, updated_params
            )
            ssfm = SSFM(potential, initial_wave_function, updated_params)
            sym_ssfm = SSFMSymmetric(potential, initial_wave_function, updated_params)

            solvers = [crank_nicolson, ssfm, sym_ssfm]

            exact_wf = exact_solver.update()

            # calculate differences as each norm for each solver
            for solvername, solver in zip(solver_names, solvers):
                current_wf = solver.update()
                for normname, norm in zip(norm_names, norms):
                    self.results[(solvername, normname)].append(
                        norm(exact_wf, current_wf)
                    )


class Saver:
    """Helper class acting as an abstraction for saving
    TimeStepper results."""

    params: Params
    results: dict
    dt_space: NDArray[np.float64]
    modes: list[list[int, int]] | NDArray[np.int32]
    coeffs: list[complex] | NDArray[np.complex128]

    def __init__(self):
        pass

    def from_helper(self, time_stepper: TimeStepper) -> None:
        self.params = time_stepper.params
        self.results = time_stepper.results
        self.dt_space = time_stepper.dt_space
        self.modes = time_stepper.modes
        self.coeffs = time_stepper.coeffs

    def write(self, file: str) -> None:
        with open(file, "wb") as f:
            pickle.dump(
                {
                    "params": self.params,
                    "results": self.results,
                    "dt_space": self.dt_space,
                    "modes": self.modes,
                    "coeffs": self.coeffs,
                },
                f,
                protocol=-1,
            )

    def read(self, file: str) -> None:
        with open(file, "rb") as f:
            loaded = pickle.load(f)

        self.params = loaded["params"]
        self.results = loaded["results"]
        self.dt_space = loaded["dt_space"]
        self.modes = loaded["modes"]
        self.coeffs = loaded["coeffs"]


def plotting(
    thing: TimeStepper | Saver, show: bool = True
) -> None | tuple[plt.Figure, Any]:
    """Helper function for plotting results of TimeStepper."""

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    plt.style.use("../../examples/JK_W.mplstyle")

    solver_names = ["cn", "ssfm", "sym_ssfm"]
    norm_names = ["sup", "l2"]

    title = r"$\Delta = \Psi_{\text{Method}}(\delta t) - \Psi_{\text{Exact}}(\delta t)$"
    x_titles = {
        "cn": "Crank-Nicolson",
        "ssfm": "Split-step Fourier",
        "sym_ssfm": "Symmetric Split-step Fourier",
    }
    y_titles = {"sup": r"$\sup|\Delta|$", "l2": r"$\iint|\Delta|^2$"}
    xlabel = r"$\delta t$ [a. u.]"

    for (i, normname), (j, solvername) in product(
        enumerate(norm_names), enumerate(solver_names)
    ):
        ax[i][j].plot(thing.dt_space, thing.results[(solvername, normname)], "o")
        ax[i][j].set_xlabel(xlabel)
        ax[i][j].set_xscale("log")
        ax[i][j].set_yscale("log")
        ax[i][j].set_ylabel(y_titles[normname])
        ax[i][j].set_title(x_titles[solvername])
        # ax[i][j].ticklabel_format(
        #     axis="y", style="sci", scilimits=(0, 0), useMathText=True
        # )

    fig.suptitle(title)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax
