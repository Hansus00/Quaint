from StationaryWaveFunc import StationaryWaveFunc
from Propagator import Propagator


class Solver:
    def __init__(self):
        raise NotImplementedError

    def __call__(
        self,
        waveFunc: StationaryWaveFunc,
        propagator: Propagator,
        deltaT: float = 1e-3,
        n: int = 1,
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class CrankNicolson(Solver):
    raise NotImplementedError


class SSFM(Solver):
    raise NotImplementedError
