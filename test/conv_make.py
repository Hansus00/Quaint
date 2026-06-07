import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from basis_convergence_overview import *
import numpy as np

r0 = np.array([50, 50])
k0 = np.array([0, 0])
sigma0 = np.array([[5, 0], [5, 0]])
Lx = 10
Ly = 10
grid_step = 0.1

params = Params(r0=r0, k0=k0, length_x=Lx, length_y=Ly, grid_step=grid_step, mass=1)


h = Helper(params)

nspace = np.arange(10, 150)

h.calculate_comparisons(nspace)

s = Saver()

s.from_helper(h)

s.write("./test/conv_data/calcs.pickle")
