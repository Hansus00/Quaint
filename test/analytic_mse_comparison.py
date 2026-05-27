'''Plots out mse of different solvers for the gaussian in a box system.'''
import numpy as np

import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Analytic import GaussianInWellSolver
from backend.Solver import CrankNicolson, SSFMSymmetric
from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket, StationaryWaveFunc
import matplotlib.pyplot as plt

N = 1000

SIZE = 60
grid_step = 1

r0 = (30, 30)
k0 = np.array([0, 0])
sigma0 = np.array([[5, 0], [0, 5]])
mass = 3e-2
delta_t = 0.001


def MSE(a: StationaryWaveFunc, b: StationaryWaveFunc):
    return np.sum((a.matrix - b.matrix) ** 2)

well = InfiniteWellPotential(SIZE, SIZE)
wf = GaussianPacket(r0, k0, sigma0, mass, SIZE, SIZE)

CN = CrankNicolson(well, wf, delta_t, grid_step)
SSFM = SSFMSymmetric(well, wf, delta_t, grid_step)
GIW = GaussianInWellSolver(k0, r0, sigma0, SIZE, mass, delta_t, grid_step)

solvers = [CN, SSFM, GIW]
wave_funcs = [CN.get_wave_function(),SSFM.get_wave_function(),GIW.get_wave_function()]

mse_cn_giw = [MSE(wave_funcs[0], wave_funcs[2])]
mse_ssfm_giw = [MSE(wave_funcs[1], wave_funcs[2])]
mse_cn_ssfm = [MSE(wave_funcs[0], wave_funcs[1])]

norm_giw = [wave_funcs[2].total_probability()]
norm_cn = [wave_funcs[0].total_probability()]
norm_ssfm = [wave_funcs[1].total_probability()]

print(
    f"""\nInitial mse: \nCN-SSFM: {mse_cn_ssfm[0]}\nCN-GIW: {mse_cn_giw[0]}\nSSFM-GIW: {mse_ssfm_giw[0]}\n
    \nInitial norm: \nGIW: {norm_giw[0]}\nCN: {norm_cn[0]}\nSSFM: {norm_ssfm[0]}\n"""
)

for i in range(N):
    for i,s in enumerate(solvers):
        wave_funcs[i] = s.update()

    mse_cn_giw.append(MSE(wave_funcs[0], wave_funcs[2]))
    mse_ssfm_giw.append(MSE(wave_funcs[1], wave_funcs[2]))
    mse_cn_ssfm.append(MSE(wave_funcs[0], wave_funcs[1]))
    norm_giw.append(wave_funcs[2].total_probability())
    norm_cn.append(wave_funcs[0].total_probability())
    norm_ssfm.append(wave_funcs[1].total_probability())



t = np.linspace(0, N * delta_t, N + 1)

fig, ax = plt.subplots(2, 3, figsize=(18, 12))

ax[0][0].plot(t, mse_cn_giw)
ax[0][0].set_title('MSE(CN, GIW)')
ax[0][1].plot(t, mse_ssfm_giw)
ax[0][1].set_title('MSE(SSFM, GIW)')
ax[0][2].plot(t, mse_cn_ssfm)
ax[0][2].set_title('MSE(CN, SSFM)')
ax[1][0].plot(t, norm_giw)
ax[1][0].set_title('NORM(GIW)')
ax[1][1].plot(t, norm_cn)
ax[1][1].set_title('NORM(CN)')
ax[1][2].plot(t, norm_ssfm)
ax[1][2].set_title('NORM(SSFM)')

for a in ax:
    for b in a:
        b.set_ylim(0,10)


plt.show()

from datetime import datetime

# run from test directory
fig.savefig(f'../pic/analytic_mse/{datetime.now()}.png',format='png')