import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import (
    InfiniteWellPotential,
    HarmonicPotential,  # should be left as is for later usage
    GaussianBumpPotential,
)
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson
import numpy as np


class CrankNicolsonEnergyTest(CrankNicolson):
    def __init__(self, potential, wave_func, delta_t=0.001):
        super().__init__(potential, wave_func, delta_t)

    def energy(self) -> float:
        """Returns expected value of the hamiltonian."""
        return np.sum(
            np.conjugate(self._wave_state_1D) * (self.H @ self._wave_state_1D)
        )


def test_energy_conservation(
    solver: CrankNicolson, N: int = 1000, eps: float = 0.01
) -> None:
    def current():
        solver.update()
        return solver.energy()

    energy = np.array([solver.energy()] + [current() for i in range(N)])

    # Expected values of hermitian operators are real
    assert np.allclose(np.imag(energy), np.zeros_like(energy))

    energy = np.real(energy)

    energy_variation = np.abs((np.max(energy) - np.min(energy))) / (
        np.abs(np.max(energy)) + np.abs(np.min(energy))
    )

    assert energy_variation < eps


if __name__ == "__main__":
    Nx = 20
    Ny = 20

    N = 2000
    eps = 1e-5

    wf = GaussianPacket(
        [10, 10], np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1, Nx, Ny
    )

    # infinite well
    V = InfiniteWellPotential(Nx, Ny)
    nc = CrankNicolsonEnergyTest(V, wf)

    test_energy_conservation(nc, N, eps)

    # gaussian bump
    V = GaussianBumpPotential(Nx, Ny, (0, 0), 2, np.array([[1, 0], [0, 1]]))
    nc = CrankNicolsonEnergyTest(V, wf)

    test_energy_conservation(nc, N, eps)

    print("Test passed")
