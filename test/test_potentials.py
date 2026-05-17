# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import *

if __name__ == "__main__":
    swp = SharpWShapedPotential(20, 20, 3, 1)
    print(swp)

    inside = PotentialInsideGrid(35, 35, 5, 5, swp)
    print(inside)

    ipw = InfiniteWellPotential(10, 10, 1)
    print(ipw)

    inside2 = PotentialInsideGrid(20, 20, 5, 5, ipw)
    print(inside2)

# %%
