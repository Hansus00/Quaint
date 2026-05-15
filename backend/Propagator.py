from StationaryWaveFunc import StationaryWaveFunc
from Potential import Potential
import numpy as np

class Propagator:
    """
    returns psi(t+Delta t) = U psi(t)
    """

    potential: Potential

    def __init__(self, potential: Potential):
        self.potential = potential

    def _step(self, wave_func: StationaryWaveFunc, delta_t: float = 1.e-3) -> StationaryWaveFunc:
        raise NotImplementedError
    
    def step_multiple(self, wave_func: StationaryWaveFunc, delta_t: float = 1.e-3, n_steps=1) -> StationaryWaveFunc:
        raise NotImplementedError

    # def __call__(
    #     self, wave_func: StationaryWaveFunc, delta_t: float = 1e-3
    # ) -> StationaryWaveFunc:
    #     raise NotImplementedError


class Cayley(Propagator):
    
    def __init__(self, potential):
        super().__init__(potential)

    def step(self, wave_func: StationaryWaveFunc, delta_t: float = 1.e-3) -> StationaryWaveFunc:
        momentum_squared = wave_func.p_x **2 + wave_func.p_y ** 2
        Hamiltonian = momentum_squared/ 2 / wave_func.mass + self.potential

        complex_factor = 1j*Hamiltonian * delta_t / 2
        factor = np.linalg.inv(1 + complex_factor)*(1 - complex_factor)

        return factor * wave_func
    
    def step_multiple(self, wave_func: StationaryWaveFunc, delta_t: float = 1.e-3, n_steps=1) -> StationaryWaveFunc:
        return self._step(wave_func, delta_t, n_steps)




class SuzukiTrotter(Propagator):
    
    def __init__(self, potential):
        super().__init__(potential)