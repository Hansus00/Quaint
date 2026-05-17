# %%
import sys
from pathlib import Path

# to call backend module from current directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.Potential import *

if __name__ == "__main__":
    ws = WShaped(20, 20, 3, 1)
    print(ws)

    inside = EmbeddedPotential(35, 35, 5, 5, ws)
    print(inside)

    ipw = InfiniteWellPotential(10, 10, 1)
    print(ipw)

    inside2 = EmbeddedPotential(20, 20, 5, 5, ipw)
    print(inside2)

    ipw2 = InfiniteWellPotential(20, 20, 1)
    matryoshka = inside2 + ipw2
    print(matryoshka)


# %%
