#!/bin/python3

from numpy.typing import NDArray
import numpy as np


class Plain:
    matrix: NDArray[np.complex128] = np.eye(1, dtype=np.complex128)

    def __init__(self):
        pass


class StationaryWaveFunc(Plain):
    pass


class Potential(Plain):
    pass


class Propagator:
    def __init__(self):
        pass

    def __call__(
        self, waveFunc: StationaryWaveFunc, potential: Potential
    ) -> StationaryWaveFunc:
        pass


class Solver:
    def __init__(self):
        pass

    def __call__(
        self,
        waveFunc: StationaryWaveFunc,
        propagator: Propagator,
        deltaT: float = 1e-3,
        n: int = 1,
    ) -> StationaryWaveFunc:
        pass


class Cayley(Propagator):
    pass


class SuzukiTrotter(Propagator):
    pass


class CrankNicolson(Solver):
    pass


class SSFM(Solver):
    pass


class AnalyticalRectangle(Solver):
    """Analytical solution for infinite rectangle potential well"""

    pass
