from StationaryWaveFunc import StationaryWaveFunc
from Potential import Potential


class Propagator:
    potential: Potential

    def __init__(self, potential: Potential):
        self.potential = potential

    def __call__(
        self, waveFunc: StationaryWaveFunc, potential: Potential
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class Cayley(Propagator):
    raise NotImplementedError


class SuzukiTrotter(Propagator):
    raise NotImplementedError
