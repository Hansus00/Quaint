# ==============================================================================
# ### --- FILE main.py --- ###
# ==============================================================================

import sys

from PyQt6.QtWidgets import (
    QApplication,
)
from main_window import MainWindow
import numpy as np


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
