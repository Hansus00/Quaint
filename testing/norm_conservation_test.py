# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import InfiniteWellPotential
from backend.StationaryWaveFunc import GaussianPacket
from backend.Solver import CrankNicolson, Constant
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("JK_W.mplstyle")
ipw = InfiniteWellPotential(128, 128, 1e6)  # Mock potential well

gauss = GaussianPacket(
    (64, 64),
    np.array([0, 0]),
    np.array([[16, 0], [0, 16]]),
    0.001,
    *ipw.matrix.shape,
)
delta_t = 0.001
cn = CrankNicolson(ipw, gauss,delta_t)

# %%
N = 2048

t_array = np.linspace(0,delta_t*N,N+1)
prob = [cn.update().total_probability() for i in range(N)]
prob = [gauss.total_probability()] + prob

plt.plot(t_array,prob,".")
plt.title(f'Total probability in time, delta_t = {delta_t}')
plt.xlabel('t')
plt.ylabel('P')
plt.ylim(0.99, 1.01)
plt.savefig(f'pic/probability/prob-in-time_delta-t={delta_t}_N={N}.png')
plt.show()
# %%
