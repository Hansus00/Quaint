import numpy as np
from typing import Callable
from numpy.typing import NDArray

from .Solver import _Solver
from .StationaryWaveFunc import StationaryWaveFunc
from .Potential import InfiniteWellPotential


class _AnalyticSolver(_Solver):
    _wave_lambda: Callable
    _grid: NDArray

    def __init__(self, potential, wave_func, delta_t=0.001, grid_step=1):
        super().__init__(potential, wave_func, delta_t, grid_step)

    def _stability_conditions(self):
        pass

    def get_wave_function(self) -> StationaryWaveFunc:
        return StationaryWaveFunc(self._wave_lambda(self._steps_evolved * self.delta_t))


class InfiniteWellSolver(_AnalyticSolver):
    def __init__(self, Nx, Ny, Lx, Ly, grid_size, mass, delta_t=0.001, grid_step=1):
        pos_1d = np.linspace(0, grid_size, grid_size // grid_step)

        self._grid = np.meshgrid(pos_1d, pos_1d, indexing="ij")
        energy = np.pi**2 / (2 * mass) * (Nx**2 / Lx**2 + Ny**2 / Ly**2)
        self._wave_lambda = (
            lambda t: 2
            / np.sqrt(Lx * Ly)
            * np.sin(Nx * np.pi / Lx * self._grid[0])
            * np.sin(Ny * np.pi / Ly * self._grid[1])
            * np.exp(1j * energy * t)
        )

        super().__init__(InfiniteWellPotential, None, delta_t, grid_step)


class GaussianPacketSolver(_AnalyticSolver):
    def __init__(self, k0, r0, a, grid_size, mass, delta_t=0.001, grid_step=1):
        pos_1d = np.linspace(0, grid_size, grid_size // grid_step)

        self._grid = np.meshgrid(pos_1d, pos_1d, indexing="ij")

        self._psi = lambda x, t: (
            np.power(2 * a**2 / np.pi / (a**4 + 4 * t**2 / mass**2), 0.25)
            * np.exp(
                1j
                * (
                    k0[0] * (x - r0[0])
                    - 0.5 * np.arctan(2 * t / mass / a**2)
                    - k0[0] ** 2 / 2 / mass * t
                )
                - (x - k0[0] * t / mass) ** 2 / (a**2 + 2j * t / mass)
            )
        )

        self._wave_lambda = lambda t: (
            self._psi(self._grid[0], t) * self._psi(self._grid[1], t)
        )

        super().__init__(InfiniteWellPotential, None, delta_t, grid_step)
