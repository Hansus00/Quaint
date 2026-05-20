# %%
from backend.Potential import InfiniteWellPotential
from backend.Solver import CrankNicolson
from backend.StationaryWaveFunc import GaussianPacket

import matplotlib.pyplot as plt
import numpy as np

# %%
V = InfiniteWellPotential(50, 50)
wf = GaussianPacket([25, 25], np.array([0, 100]), np.array([[2, 0], [0, 2]]), 1e-2, 50, 50)
nc = CrankNicolson(V, wf)

# %%
plt.imshow(nc.get_wave_function().matrix.__abs__(),origin='lower')
plt.show()

# %%
plt.imshow(nc.update(10).matrix.__abs__(),origin='lower')
plt.show()
# %%
def calc_ev_x(wavefunc):
    x = np.array([[[i,j] for i in range(50)] for j in range(50)])
    return np.einsum("ij,ijk,ij->k",np.conjugate(wavefunc), x, wavefunc)

n = 40
x_array = np.array([calc_ev_x(wf.matrix)] + [calc_ev_x(nc.update().matrix) for i in range(n)])

# %%
plt.plot(*x_array.T,'bo')
plt.plot(*x_array[0].T,'ro')
# %%
x_array