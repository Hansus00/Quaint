import sys

from PyQt6.QtWidgets import (
    QApplication,
)
from wave_function_simulator import WaveFunctionSimulator
import numpy as np


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    app = QApplication(sys.argv)
    window = WaveFunctionSimulator(size_x=50, size_y=60)
    window.show()
    sys.exit(app.exec())
