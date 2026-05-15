import sys

from PyQt6.QtWidgets import (
    QApplication,
)
from wave_function_simulator import WaveFunctionSimulator

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveFunctionSimulator()
    window.show()
    sys.exit(app.exec())
