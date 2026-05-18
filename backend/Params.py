# %%
from enum import Enum
from dataclasses import dataclass, asdict, field
import json
from numpy.typing import NDArray
import numpy as np


class WellType(str, Enum):
    W_SHAPED = "w-shaped"
    INFINITE_WELL = "infiniteWell"
    MATRYOSHKA = "matryoshka"
    NONE = "none"


class SolverType(str, Enum):
    CN = "cn"
    SSFM = "ssfm"


@dataclass
class Params:
    size_x: int = 128
    size_y: int = 128

    well_type: WellType = (
        WellType.INFINITE_WELL
    )  # TODO: maybe save entire Potential.matrix?
    well_height: float = 1e6

    solver: SolverType = SolverType.SSFM
    r0: list[float] = field(default_factory=lambda: [64, 64])
    k0: list[float] = field(default_factory=lambda: [1, 0])
    sigma0: list[list[float]] = field(default_factory=lambda: [[16, 0], [0, 16]])
    mass: float = 2e-3
    delta_n: int = 32
    delta_t: float = 1e-3
    steps_max: int = 4

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
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in data}


if __name__ == "__main__":
    p = Params()
    p.write("b.json")

    d = Params()
    d.read("aa.json")
    print(d)

# print(Params())
# %%
