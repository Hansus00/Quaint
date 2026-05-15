from StationaryWaveFunc import StationaryWaveFunc
from Potential import Potential
import numpy as np
import numpy.linalg as la


class Solver:
    potential: Potential

    def __init__(self, potential: Potential):
        self.potential = potential

    def __call__(
        self,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        n_steps: int = 1,
    ) -> StationaryWaveFunc:
        raise NotImplementedError


class CrankNicolson(Solver):
    def __init__(self, potential: Potential):
        super().__init__(potential)

    def __call__(
        self,
        wave_func: StationaryWaveFunc,
        delta_t: float = 1e-3,
        n_steps: int = 1,
    ) -> StationaryWaveFunc:
        # FIXME: zrobić
        A = np.zeros(self.potential.matrix.shape)
        B = np.zeros(self.potential.matrix.shape)

        return StationaryWaveFunc(la.inv(A) @ B @ wave_func.matrix)


class Constant(Solver):
    def __init__(self, potential: Potential):
        super().__init__(potential)

    def __call__(
        self, wave_func: StationaryWaveFunc, delta_t: float = 1e3, n_steps: int = 1
    ) -> StationaryWaveFunc:
        return wave_func


class SSFM(Solver):
    raise NotImplementedError
