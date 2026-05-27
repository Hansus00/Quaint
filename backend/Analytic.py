import numpy as np
from typing import Callable
from numpy.typing import NDArray

from .Solver import _Solver
from .StationaryWaveFunc import StationaryWaveFunc
from .Potential import InfiniteWellPotential
from .Potential import Potential

from mpmath import jtheta


class _AnalyticSolver(_Solver):
    delta_t: float
    _wave_lambda: Callable
    _grid: NDArray
    _mass: float
    _potential: Potential

    def __init__(
        self,
        potential: Potential,
        wave_func: Callable,
        grid_size: float,
        mass: float,
        delta_t: float = 0.001,
        grid_step: float = 1,
    ):
        self.delta_t = delta_t
        self._mass = mass
        self._wave_lambda = wave_func
        self._potential = potential

        self._pos_1d = np.linspace(0, grid_size, grid_size // grid_step)
        self._grid = np.meshgrid(self._pos_1d, self._pos_1d, indexing="ij")

    def _stability_conditions(self):
        pass  # assumes correct solution

    def update(self, n_step=1):
        self._steps_evolved += n_step
        return self.get_wave_function()  # time parameter is enough

    def get_wave_function(self) -> StationaryWaveFunc:
        return StationaryWaveFunc(
            self._wave_lambda(self._steps_evolved * self.delta_t), self._mass
        )  # simple evaluation at current time


class InfiniteWellSolver(_AnalyticSolver):
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        grid_size: float,
        mass: float,
        delta_t: float = 0.001,
        grid_step: float = 1,
    ):
        """Infinite well stationary solution in the (0,0) corner in the [0, Lx] x [0, Ly]
        region. The solution has the (Nx, Ny) index (it is the Nx'th mode in the x direction
        and the Ny'th mode in the y direction). The potential field of this class
        does not correspond to the actual potential this soulution obeys."""
        # wf is only inside the well
        well_mask = (0 <= self._grid[0] <= Lx) * (0 <= self._grid[1] <= Ly)

        # source: https://en.wikipedia.org/wiki/Particle_in_a_box#Higher-dimensional_boxes
        energy = np.pi**2 / (2 * mass) * (Nx**2 / Lx**2 + Ny**2 / Ly**2)

        wave_func = (
            lambda t: well_mask
            * 2
            / np.sqrt(Lx * Ly)
            * np.sin(Nx * np.pi / Lx * self._grid[0])
            * np.sin(Ny * np.pi / Ly * self._grid[1])
            * np.exp(1j * energy * t)
        )
        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )


class GaussianPacketSolver(_AnalyticSolver):
    def __init__(
        self,
        k0: NDArray[np.float64],
        r0: NDArray[np.float64],
        sigma0: NDArray[np.float64],
        grid_size: float,
        mass: float,
        delta_t: float = 0.001,
        grid_step=1,
    ):
        """Warning: Unlike GaussianPacket this class does not accept general
        sigma. Instead, quadratic mean of the diagonal entries is taken as
        the standard deviation"""
        a = np.sqrt(np.hypot(*np.diag(sigma0)) / np.sqrt(2))
        # source: https://en.wikipedia.org/wiki/Wave_packet#The_2D_case
        psi = lambda i, t: (
            np.power(2 * a**2 / np.pi / (a**4 + 4 * t**2 / mass**2), 0.25)
            * np.exp(
                1j
                * (
                    k0[i] * (self._grid[i] - r0[i])
                    - 0.5 * np.arctan(2 * t / mass / a**2)
                    - k0[i] ** 2 / 2 / mass * t
                )
                - (self._grid[i] - r0[i] - k0[i] * t / mass) ** 2
                / (a**2 + 2j * t / mass)
            )
        )

        wave_func = lambda t: (psi(0, t) * psi(1, t))

        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )

    def ev_energy(self) -> float:
        return 1  # for testing purposes


class GaussianInWellSolver(_AnalyticSolver):
    def __init__(
        self,
        k0: NDArray[np.float64],
        r0: NDArray[np.float64],
        sigma0: NDArray[np.float64],
        grid_size: float,
        mass: float,
        delta_t: float = 0.001,
        grid_step=1,
    ):
        """Warning: this is a clunker suggested solution and
        partially checked clanker code.
        sigma0 takes ndarray but only sigma0[0][0] is used as
        a standard deviation ** 2"""
        sigma0 = sigma0[0][0]

        self._jtheta_numpied = np.frompyfunc(
            lambda z, q: np.complex128(complex(jtheta(3, z, q))), 2, 1
        )

        gamma = lambda t: 1 / (4 * sigma0**2 * (1 + 1j * t) / (2 * mass * sigma0**2))

        q = lambda t: np.exp(-4 * gamma(t) * grid_size**2)

        z_plus = (
            lambda i, t: -2j
            * gamma(t)
            * grid_size
            * (self._pos_1d - r0[i] - k0[i] * t / mass)
            - k0[i] * grid_size
        )
        z_minus = (
            lambda i, t: -2j
            * gamma(t)
            * grid_size
            * (self._pos_1d + r0[i] + k0[i] * t / mass)
            + k0[i] * grid_size
        )

        env_plus = lambda i, t: np.exp(
            -gamma(t) * (self._pos_1d - r0[i] - k0[i] * t / mass) ** 2
            + 1j * k0[i] * (self._pos_1d - k0[i] * t / (2 * mass))
        )

        env_minus = lambda i, t: np.exp(
            -gamma(t) * (self._pos_1d + r0[i] + k0[i] * t / mass) ** 2
            + 1j * k0[i] * (self._pos_1d + k0[i] * t / (2 * mass))
        )

        theta_plus = lambda i, t: self._jtheta_numpied(z_plus(i, t), q(t))
        theta_minus = lambda i, t: self._jtheta_numpied(z_minus(i, t), q(t))

        psi_1d = lambda i, t: (
            env_plus(i, t) * theta_plus(i, t) - env_minus(i, t) * theta_minus(i, t)
        )

        wave_func = lambda t: np.outer(psi_1d(0, t), psi_1d(1, t))

        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )

        norm = np.sqrt(np.sum(np.abs(wave_func(0)) ** 2))

        wave_func = lambda t: np.outer(psi_1d(0, t), psi_1d(1, t)) / norm
        
        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )

    def ev_energy(self) -> float:
        return 1  # for testing purposes
