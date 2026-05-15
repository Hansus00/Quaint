from numpy.typing import NDArray
import numpy as np


class Plain:
    matrix: NDArray[np.complex128] = np.eye(1, dtype=np.complex128)

    def __init__(self):
        raise NotImplementedError


class StationaryWaveFunc(Plain):
    def __init__(self):
        super().__init__()


class Potential(Plain):
    def __init__(self):
        super().__init__()


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
    
    def __init__(self):
        super().__init__()
