# ==============================================================================
# ### --- FILE main.py --- ###
# ==============================================================================

import sys

import numpy as np
from frontend.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
import frontend.LoggerTools as LoggerTools

if __name__ == "__main__":
    LoggerTools.configLogger(
        LoggerTools.INFO
    )  # TODO: change it with flags as --dbg --info --quiet --verbose etc.
    # Ensure full numpy print outputs for debugging purposes if needed
    np.set_printoptions(threshold=np.inf)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
