from backend.Solver import *

class CrankNicolsonTest(CrankNicolson):
    def __init__(self, potential, wave_func, delta_t=0.001):
        super().__init__(potential, wave_func, delta_t)

    def energy(self):
        return np.sum(
            np.conjugate(self._wave_state_1D) * (self.H @ self._wave_state_1D)
        )
