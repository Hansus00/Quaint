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

    well_type: WellType = WellType.INFINITE_WELL
    well_height: float = 1e6

    solver: SolverType = SolverType.CN
    r0: tuple[int, int] = (64, 64)
    k0: NDArray[np.float64] = (np.array([1, 0]).tolist(),)
    sigma0: list = field(default_factory=lambda: np.array([[16, 0], [0, 16]]).tolist())
    mass: float = 2e-3
    delta_n: int = 32
    delta_t: int = 1e-3
    steps_max: int = 4

    @classmethod
    def _from_dict(self, data: dict):
        if "well-type" in data:
            data["well_type"] = WellType(data["well_type"])
        if "solver" in data:
            data["solver"] = SolverType(data["solver"])
        return self(**data)

    def read(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        new_params = Params._from_dict(raw_data)
        self.__dict__.update(new_params.__dict__)

    def write(self, filepath: str) -> None:
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
