from enum import Enum
from dataclasses import dataclass, asdict, field
import json
from numpy.typing import NDArray
import numpy as np


class WellType(str, Enum):
    W_SHAPED = "w-shaped"
    INFINITE_WELL = "infiniteWell"
    MATRYOSHKA = "matryoshka"
    SLAB = "slab"
    DOUBLE_SLIT = "double_slit"
    NONE = "none"


class SolverType(str, Enum):
    CN = "cn"
    SSFM = "ssfm"
    SYM_SSFM = "sym_ssfm"


@dataclass
class Params:
    size_x: int = 128
    size_y: int = 128

    well_type: WellType = (
        WellType.INFINITE_WELL
    )  # TODO: maybe save entire Potential.matrix?
    well_height: float = 1e6
    inside_wall_height: float = 1e6  # height of whatever is inside

    solver: SolverType = SolverType.SSFM
    r0: tuple[int, int] = field(default_factory=lambda: (64, 64))
    k0: NDArray[np.float64] = field(default_factory=lambda: np.array([0.1, 0]))
    sigma0: NDArray[np.float64] = field(
        default_factory=lambda: np.array([[16, 0], [0, 16]])
    )
    mass: float = 1e-3
    delta_n: int = 32  # steps per update
    delta_t: float = 1e-4  # time step per update
    grid_step: float = 1
    updates_max: int = 4  # how many updates, each one changes by delta_n

    @classmethod
    def _from_dict(cls, data: dict):
        if "well_type" in data:
            data["well_type"] = WellType(data["well_type"])
        if "solver" in data:
            data["solver"] = SolverType(data["solver"])
        return cls(**data)

    def read(self, filepath: str) -> None:
        """Read simulation parameters from file located at filepath"""
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        new_params = Params._from_dict(raw_data)
        self.__dict__.update(new_params.__dict__)

    def write(self, filepath: str) -> None:
        """Write simulation parameters into file located at filepath"""
        p_dict = asdict(self, dict_factory=_enum_dict_factory)
        with open(filepath, "w") as f:
            json.dump(p_dict, f, indent=4)


def _enum_dict_factory(data):
    return {
        k: (
            v.value
            if isinstance(v, Enum)
            else v.tolist() if isinstance(v, np.ndarray) else v
        )
        for k, v in data
    }
