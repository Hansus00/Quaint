import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import InfiniteWellBasisSolver
from backend.StationaryWaveFunc import GaussianPacket

wf = GaussianPacket(
    (50, 50),
    np.array([0, 0]),
    sigma0=[[5, 0], [0, 5]],
    mass=2e-3,
    size_x=100,
    size_y=100,
)

N = 150
Nx, Ny = N, N
delta_t = 0.001

bs = InfiniteWellBasisSolver(wf, Nx, Ny, delta_t)


def diff_against_actual_wf(N):
    N = int(N)
    return (
        np.einsum("ij,ijkl->kl", bs._coeffs[:N, :N], bs._basis[:N, :N, :, :])
        - wf.matrix
    )


def diff_subsequent(N):
    N = int(N)
    return np.einsum(
        "i,ikl->kl", bs._coeffs[:N, N], bs._basis[:N, N, :, :]
    ) + np.einsum("i,ikl->kl", bs._coeffs[N, : N + 1], bs._basis[N, : N + 1, :, :])


Nspace = np.linspace(1, N - 1, N - 1)

fig, ax = plt.subplots(2, 3, figsize=(24, 16))

ax[0][0].plot(Nspace, [np.sum(np.abs(diff_against_actual_wf(n))**2) for n in Nspace])
ax[0][1].plot(Nspace, [np.sum(np.abs(diff_subsequent(n))**2) for n in Nspace])
im1 = ax[1][0].imshow(np.abs(wf.matrix)**2)
im2 = ax[1][1].imshow(np.abs(bs.get_wave_function().matrix)**2)
im3 = ax[1][2].imshow(np.abs(wf.matrix-bs.get_wave_function().matrix)**2)
fig.colorbar(im1)
fig.colorbar(im2)
fig.colorbar(im3)

print(np.sum(np.abs(wf.matrix)**2))
print(np.sum(np.abs(bs.get_wave_function().matrix)**2))
print(np.sum(np.abs(wf.matrix-bs.get_wave_function().matrix)**2))
print(np.sum(np.abs(bs._coeffs)**2))

plt.show()