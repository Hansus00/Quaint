import numpy as np
from typing import Callable
from numpy.typing import NDArray
from itertools import product

from .Solver import _Solver
from .StationaryWaveFunc import StationaryWaveFunc
from .Potential import InfiniteWellPotential
from .Potential import Potential


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
        grid_size: int,
        mass: float,
        delta_t: float = 0.001,
        grid_step: float = 1,
    ):
        self.delta_t = delta_t
        self._mass = mass
        self._wave_lambda = wave_func
        self._potential = potential

        self._pos_1d = np.arange(1, grid_size + 1) * grid_step
        self._grid = np.meshgrid(self._pos_1d, self._pos_1d, indexing="ij")

    def update(self, n_step=1):
        self._steps_evolved += n_step
        return self.get_wave_function()  # time parameter is enough

    def get_wave_function(self) -> StationaryWaveFunc:
        return StationaryWaveFunc(
            self._wave_lambda(self._steps_evolved * self.delta_t)
        )  # simple evaluation at current time


class FreeGaussianSolver(_AnalyticSolver):
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
        """Warning: Unlike GaussianPacket this class does not utilize general
        sigma. Instead, sigma0[0][0] is taken as the standard deviation."""
        a = sigma0[0][0]
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

        self._energy = np.hypot(k0) ** 2 / (2 * mass)

        super().__init__(
            InfiniteWellPotential, wave_func, grid_size, mass, delta_t, grid_step
        )

    def ev_energy(self) -> float:
        """Expected value of hamiltonian

        Returns:
            float: $\frac{k0**2}{2m}$
        """
        return self._energy


class InfiniteWellBasisSolver(_Solver):
    def __init__(
        self,
        wave_func: StationaryWaveFunc,
        mass: float,
        Nx: int = 30,
        Ny: int = 30,
        delta_t: float = 0.001,
        grid_step: float = 1,
        memory_saving: bool = False,
    ):
        """Creates analytic solver for an arbitrary initial wavefunction
        in an infinite potential well. Evolution is done by basis summation.

        Args:
            wave_func (StationaryWaveFunc): Initial wave function
            mass (float): Mass
            Nx (int, optional): Maximal x mode considered. Defaults to 30.
            Ny (int, optional): Maximal y mode considered. Defaults to 30.
            delta_t (float, optional): Time step. Defaults to 0.001.
            grid_step (float, optional): Grid spacing. Defaults to 1.
        """
        self.delta_t = delta_t
        self.mass = mass

        sizex = wave_func.matrix.shape[0]
        sizey = wave_func.matrix.shape[1]

        Lx = (sizex + 1) * grid_step
        Ly = (sizey + 1) * grid_step

        pos1dx = np.arange(1, sizex + 1) * grid_step
        pos1dy = np.arange(1, sizey + 1) * grid_step

        if not memory_saving:
            nxarray = np.linspace(1, Nx, Nx)
            nyarray = np.linspace(1, Ny, Ny)

            nxspace = np.einsum("i,j->ij", nxarray, (np.pi / Lx) * pos1dx)
            nyspace = np.einsum("i,j->ij", nyarray, (np.pi / Ly) * pos1dy)

            self._basis = np.einsum(
                "ij,kl->ikjl",
                np.sqrt(2 / Lx) * np.sin(nxspace),
                np.sqrt(2 / Ly) * np.sin(nyspace),
            )

            self._coeffs = (
                np.einsum("ijkl,kl->ij", np.conj(self._basis), wave_func.matrix)
                * grid_step**2
            )

            energy1dx = np.pi**2 * nxarray**2 / (2 * mass * Lx**2)
            energy1dy = np.pi**2 * nyarray**2 / (2 * mass * Ly**2)

            self._energyspace = np.add.outer(energy1dx, energy1dy)

            self._wave_lambda = lambda t: np.einsum(
                "ij,ijkl->kl",
                self._coeffs * np.exp(-1j * self._energyspace * t),
                self._basis,
            )
        else:
            nxarray = np.linspace(0, Nx - 1, Nx)
            nyarray = np.linspace(0, Ny - 1, Ny)

            self._xspace, self._yspace = np.meshgrid(
                (np.pi / Lx) * pos1dx, (np.pi / Ly) * pos1dy, indexing="ij"
            )

            def basis(nx: int, ny: int) -> np.NDArray[np.float64]:
                """Normalized"""
                return (
                    np.sqrt(2 / Lx)
                    * np.sqrt(2 / Ly)
                    * np.sin((nx + 1) * self._xspace)
                    * np.sin((ny + 1) * self._yspace)
                )

            self._basis = basis

            self._coeffs = np.array(
                [
                    [
                        np.sum(np.conj(self._basis(nx, ny)) * wave_func.matrix)
                        * grid_step**2
                        for nx in nxarray
                    ]
                    for ny in nyarray
                ],
                dtype=np.complex128,
            )
            energy1dx = np.pi**2 * nxarray**2 / (2 * mass * Lx**2)
            energy1dy = np.pi**2 * nyarray**2 / (2 * mass * Ly**2)

            self._energyspace = np.add.outer(energy1dx, energy1dy)

            def result(t: float) -> np.NDArray[np.complex128]:
                res = np.zeros_like(self._xspace, dtype=np.complex128)
                for nx, ny in product(nxarray, nyarray):
                    res += (
                        self._coeffs[nx, ny]
                        * np.exp(-1j * self._energyspace[nx, ny] * t)
                        * self._basis(nx, ny)
                    )
                return res

            self._wave_lambda = result

    def update(self, n_step=1):
        self._steps_evolved += n_step
        return self.get_wave_function()  # time parameter is enough

    def get_wave_function(self) -> StationaryWaveFunc:
        return StationaryWaveFunc(
            self._wave_lambda(self._steps_evolved * self.delta_t)
        )  # simple evaluation at current time

    def ev_energy(self):
        return np.einsum("ij,ij", np.abs(self._coeffs) ** 2, self._energyspace)


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

        super().__init__(
            InfiniteWellPotential, None, grid_size, mass, delta_t, grid_step
        )

        # wf is only inside the well
        well_mask = (
            (0 <= self._grid[0])
            * (self._grid[0] <= Lx)
            * (0 <= self._grid[1])
            * (self._grid[1] <= Ly)
        )

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

    def ev_energy(self) -> float:
        return 1  # for testing purposes
