from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import json
from numpy.typing import NDArray
import numpy as np


class PotentialType(str, Enum):
    """
    Use premade potential or custom
    """

    INFINITE_WELL = "infiniteWell"  # default, every potential is inside it
    W_SHAPED = "w-shaped"
    MATRYOSHKA = "matryoshka"
    SLAB = "slab"
    DOUBLE_SLIT = "double_slit"
    CUSTOM = "custom"


class SolverType(str, Enum):
    CN = "cn"
    SSFM = "ssfm"
    SYM_SSFM = "sym_ssfm"
    ANALYTIC_GAUSSIAN = "analytic_gaussian"
    CONSTANT = "constant"


@dataclass
class Params:
    """
    length_i = N_i * grid_step, where N_i is grid size
    """

    length_x: float = 64.0
    length_y: float = 64.0
    grid_step: float = 0.4

    solver: SolverType = SolverType.SSFM

    r0: Tuple[float, float] = field(default_factory=lambda: (32.0, 32.0))
    k0: NDArray[np.float64] = field(default_factory=lambda: np.array([1.5, 0]))
    sigma0: NDArray[np.float64] = field(
        default_factory=lambda: np.array([[16, 0], [0, 16]])
    )
    mass: float = 1e-3
    delta_t: float = 1e-4
    T_tot: float = 0.05  # total time of simulation

    potential_type: PotentialType = PotentialType.INFINITE_WELL
    well_height: float = 1e6
    potential_matrix: Optional[NDArray[np.float64]] = (
        None  # should be the last parameter, as it it the biggest
    )

    @property
    def grid_size_x(self) -> int:
        return int(self.length_x / self.grid_step)

    @property
    def grid_size_y(self) -> int:
        return int(self.length_y / self.grid_step)

    @property
    def dx(self) -> float:
        return self.grid_step

    @property
    def dy(self) -> float:
        return self.grid_step

    @property
    def r0_grid(self) -> Tuple[int, int]:
        return (int(self.r0[0] / self.grid_step), int(self.r0[1] / self.grid_step))

    @property
    def sigma0_grid(self) -> NDArray[np.float64]:
        return self.sigma0 / (self.grid_step**2)

    @property
    def k0_grid(self) -> NDArray[np.float64]:
        return self.k0 * self.grid_step

    @property
    def n_steps(self) -> int:
        return int(self.T_tot / self.delta_t)

    @classmethod
    def _from_dict(cls, data: dict):
        if "potential_type" in data:
            data["potential_type"] = PotentialType(data["potential_type"])
        if "solver" in data:
            data["solver"] = SolverType(data["solver"])

        # Convert lists back to numpy arrays
        if "k0" in data:
            data["k0"] = np.array(data["k0"], dtype=np.float64)
        if "sigma0" in data:
            data["sigma0"] = np.array(data["sigma0"], dtype=np.float64)
        if "potential_matrix" in data and data["potential_matrix"] is not None:
            data["potential_matrix"] = np.array(
                data["potential_matrix"], dtype=np.float64
            )
        return cls(**data)

    def read(self, filepath: str) -> None:
        """Read simulation parameters from file located at filepath"""
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        new_params = Params._from_dict(raw_data)
        self.__dict__.update(new_params.__dict__)

    def write(self, filepath: str) -> None:
        """Write simulation parameters into file located at filepath"""
        if self.potential_matrix is not None:
            self.potential_type = PotentialType.CUSTOM  # as it has been changed by user
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
