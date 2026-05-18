# ==============================================================================
# ### --- FILE frontend/simulation_thread.py --- ###
# ==============================================================================

from typing import Any
from PyQt6.QtCore import QThread, pyqtSignal

class SimulationThread(QThread):
    """
    Worker thread for performing the physics simulation calculations without freezing the UI.
    Delegates heavy matrix operations to a separate processor thread.
    """
    
    # --- Class Fields ---
    
    # Emits: (wave_frames: list)
    calculation_finished = pyqtSignal(list)
    
    simulation: Any
    total_frames: int

    def __init__(self, simulation_instance: Any, total_frames: int) -> None:
        """
        Initializes the calculation worker thread.

        Args:
            simulation_instance (Any): The initialized backend solver (e.g., CrankNicolson, Constant).
            total_frames (int): The total number of simulation steps to pre-calculate.
        """
        super().__init__()
        self.simulation = simulation_instance
        self.total_frames = total_frames

    def run(self) -> None:
        """
        Performs the simulation calculations in a separate thread and emits 
        the aggregated results (list of frames) to the main UI once completed.
        """
        wave_frames = []
        
        # Append the initial state (t = 0)
        wave_frames.append(self.simulation.get_wave_function())

        # Iteratively calculate the subsequent time steps
        for _ in range(1, self.total_frames):
            self.simulation.step()
            wave_frames.append(self.simulation.get_wave_function())

        # Dispatch the payload back to the main thread
        self.calculation_finished.emit(wave_frames)