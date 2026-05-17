# ==============================================================================
# ### --- FILE frontend/simulation_worker.py --- ###
# ==============================================================================

from typing import Any
from PyQt6.QtCore import QThread, pyqtSignal

class SimulationThread(QThread):
    """
    Worker thread for performing the physics simulation calculations without freezing the UI.
    """
    
    calculation_finished = pyqtSignal(list)

    def __init__(self, simulation_instance: Any, total_frames: int) -> None:
        super().__init__()
        self.simulation = simulation_instance
        self.total_frames = total_frames

    def run(self) -> None:
        """Performs the simulation calculations in a separate thread and emits the results when done."""
        wave_frames = []
        
        wave_frames.append(self.simulation.get_wave_function())

        for _ in range(1, self.total_frames):
            self.simulation.step()
            wave_frames.append(self.simulation.get_wave_function())

        self.calculation_finished.emit(wave_frames)