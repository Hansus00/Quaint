import numpy as np
import matplotlib.pyplot as plt
from itertools import pairwise, product
from numpy.typing import NDArray
import pickle

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import InfiniteWellBasisSolver
from backend.StationaryWaveFunc import GaussianPacket, StationaryWaveFunc
from backend.Params import Params

# norm functions


def l2(A: np.NDArray[np.complex128], dx: float, dy: float) -> float:
    return np.sum(np.abs(A) ** 2) * dx * dy


def sup(A: np.NDArray[np.complex128]) -> float:
    return np.max(np.abs(A))


# main class


class Helper:
    coeffs: np.NDArray[np.complex128]
    basis: np.NDArray[np.float64]
    reference_wf: np.NDArray[np.complex128]
    params: Params
    arrays: dict

    def __init__(self, params: Params):
        self.params = params
        self.reference_wf = GaussianPacket(
            r0=params.r0,
            k0=params.k0,
            sigma0=params.sigma0_grid,
            size_x=params.grid_size_x,
            size_y=params.grid_size_y,
        ).matrix

    def approximator(self, nspace: np.NDArray[np.int32], memory_saving: bool = False):
        starting_n = nspace[0]

        if not memory_saving:
            current_wf = np.einsum(
                "ij,ijkl->kl",
                self.coeffs[:starting_n, :starting_n],
                self.basis[:starting_n, :starting_n, :, :],
            )

            for n in nspace:
                ny_const = self.coeffs[:n, n].flatten()
                nx_const = self.coeffs[n, :n].flatten()
                corner = self.coeffs[n, n]

                current_wf += (
                    np.einsum("i,ikl->kl", ny_const, self.basis[:n, n, :, :])
                    + np.einsum("j,jkl->kl", nx_const, self.basis[n, :n, :, :])
                    + corner * self.basis[n, n, :, :]
                )

                yield current_wf
        else:
            current_wf = np.zeros(
                (self.params.grid_size_x, self.params.grid_size_y), dtype=np.complex128
            )

            for nx, ny in product(np.arange(starting_n), np.arange(starting_n)):
                current_wf += self.coeffs[nx, ny] * self.basis(nx, ny)

            for N in nspace:
                for nx, ny in product(np.arange(N), [N]):
                    current_wf += self.coeffs[nx, ny] * self.basis(nx, ny)
                for nx, ny in product([N], np.arange(N)):
                    current_wf += self.coeffs[nx, ny] * self.basis(nx, ny)
                current_wf += self.coeffs[N, N] * self.basis(N, N)

                yield current_wf

    def calculate_comparisons(
        self, nspace: np.NDArray[np.int32], memory_saving: bool = False
    ) -> None:
        if not np.all(np.diff(nspace) == 1) or int(nspace[0]) != nspace[0]:
            raise ValueError("nspace is not a proper range")

        self.nspace = np.array(nspace, dtype=np.int32)
        augnspace = np.concat([nspace, np.array([nspace[-1] + 1])], dtype=np.int32)

        Nmax = np.max(augnspace) + 1

        solver = InfiniteWellBasisSolver(
            StationaryWaveFunc(self.reference_wf),
            self.params.mass,
            Nx=Nmax,
            Ny=Nmax,
            grid_step=self.params.dx,
            memory_saving=memory_saving,
        )

        self.coeffs = solver._coeffs
        self.basis = solver._basis

        mse_array = []
        sup_array = []
        mse_delta_array = []
        sup_delta_array = []

        for approx, next_approx in pairwise(
            self.approximator(augnspace, memory_saving)
        ):
            mse_array.append(
                l2(self.reference_wf - approx, self.params.dx, self.params.dy)
            )
            sup_array.append(sup(self.reference_wf - approx))
            mse_delta_array.append(
                l2(approx - next_approx, self.params.dx, self.params.dy)
            )
            sup_delta_array.append(sup(approx - next_approx))

        self.arrays = {
            "mse": mse_array,
            "sup": sup_array,
            "mse_delta": mse_delta_array,
            "sup_delta": sup_delta_array,
        }


class Saver:
    params: Params
    arrays: dict
    nspace: np.NDArray[np.int32]

    def __init__(self):
        pass

    def from_helper(self, helper: Helper) -> None:
        self.params = helper.params
        self.arrays = helper.arrays
        self.nspace = helper.nspace

    def write(self, file: str) -> None:
        with open(file, "wb") as f:
            pickle.dump(
                {"params": self.params, "arrays": self.arrays, "nspace": self.nspace},
                f,
                protocol=-1,
            )

    def read(self, file: str) -> None:
        with open(file, "rb") as f:
            loaded = pickle.load(f)

        self.params = loaded["params"]
        self.arrays = loaded["arrays"]
        self.nspace = loaded["nspace"]


def plotting(thing: Helper | Saver, show: bool = True):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    keys = [["mse", "sup"], ["mse_delta", "sup_delta"]]
    titles = [
        [
            r"$\int|\sum_{nx,ny}^N \phi_{nx,ny} - \Psi|^2$",
            r"$\sup|\sum_{nx,ny}^N \phi_{nx,ny} - \Psi|$",
        ],
        [
            r"$\int|\sum_{nx,ny}^{N+1} \phi_{nx,ny} - \sum_{nx,ny}^N \phi_{nx,ny}|^2$",
            r"$\sup|\sum_{nx,ny}^{N+1} \phi_{nx,ny} - \sum_{nx,ny}^N \phi_{nx,ny}|$",
        ],
    ]

    for i, j in product([0, 1], [0, 1]):
        ax[i][j].plot(thing.nspace, thing.arrays[keys[i][j]], "o")
        ax[i][j].set_xlabel("$N$")
        ax[i][j].set_title(titles[i][j])

    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax
