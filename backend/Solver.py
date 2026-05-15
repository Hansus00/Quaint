from StationaryWaveFunc import StationaryWaveFunc
from Propagator import Propagator


class Solver:
    propagator: Propagator

    def __init__(self, propagator: Propagator):
        self.propagator = propagator

    def __call__(
        self,
        waveFunc: StationaryWaveFunc,
        deltaT: float = 1e-3,
        n: int = 1,
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class CrankNicolson(Solver):
    raise NotImplementedError


class SSFM(Solver):
    raise NotImplementedError
