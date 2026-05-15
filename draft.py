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
    def __init__(self):
        super().__init__()


class SuzukiTrotter(Propagator):
    def __init__(self):
        super().__init__()


class CrankNicolson(Solver):
    def __init__(self):
        super().__init__()


class SSFM(Solver):
    def __init__(self):
        super().__init__()


class AnalyticalRectangle(Solver):
    """Analytical solution for infinite rectangle potential well"""

    raise NotImplementedError
