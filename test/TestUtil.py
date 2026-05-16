from backend.Solver import *


class CrankNicolsonTest(CrankNicolson):
    def __init__(self, potential, wave_func, delta_t=0.001):
        super().__init__(potential, wave_func, delta_t)

    def energy(self) -> float:
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

    assert np.allclose(np.imag(energy), np.zeros_like(energy))

    energy = np.real(energy)

    energy_variation = np.abs((np.max(energy) - np.min(energy))) / (
        np.abs(np.max(energy)) + np.abs(np.min(energy))
    )

    assert energy_variation < eps
