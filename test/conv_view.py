import sys
from pathlib import Path

# to call backend module from current directory
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from basis_convergence_overview import *

s = Saver()

s.read("./test/conv_data/results.pickle")

plotting(s)
