from StationaryWaveFunc import StationaryWaveFunc
from Potential import Potential


class Propagator:
    """
    returns psi(t+Delta t) = U psi(t)
    """

    potential: Potential

    def __init__(self, potential: Potential):
        self.potential = potential

    def __call__(
        self, wave_func: StationaryWaveFunc, delta_t: float = 1e-3
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class Cayley(Propagator):
    raise NotImplementedError


class SuzukiTrotter(Propagator):
    raise NotImplementedError
