# ==============================================================================
# ### --- FILE main.py --- ###
# ==============================================================================

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    # Ensure full numpy print outputs for debugging purposes if needed
    np.set_printoptions(threshold=np.inf)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())