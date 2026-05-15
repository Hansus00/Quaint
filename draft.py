from numpy.typing import NDArray
import numpy as np


class StationaryWaveFunc:
    raise NotImplementedError


class Potential:
    raise NotImplementedError


class Propagator:
    def __init__(self):
        raise NotImplementedError

    def __call__(
        self, waveFunc: StationaryWaveFunc, potential: Potential
    ) -> StationaryWaveFunc:
        raise NotImplementedError


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


class Cayley(Propagator):
    raise NotImplementedError


class SuzukiTrotter(Propagator):
    raise NotImplementedError


class CrankNicolson(Solver):
    raise NotImplementedError


class SSFM(Solver):
    raise NotImplementedError


class AnalyticalRectangle(Solver):
    """Analytical solution for infinite rectangle potential well"""

    raise NotImplementedError
