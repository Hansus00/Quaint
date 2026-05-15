from StationaryWaveFunc import StationaryWaveFunc
from Propagator import Propagator


class Solver:
    propagator: Propagator

    def __init__(self, propagator: Propagator):
        self.propagator = propagator

    def __call__(
        self,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        n_steps: int = 1,
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class CrankNicolson(Solver):
    raise NotImplementedError


class SSFM(Solver):
    raise NotImplementedError
