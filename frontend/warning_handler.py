# ==============================================================================
# ### --- FILE frontend/warning_handler.py --- ###
# ==============================================================================

import logging


class WarningCaptureHandler(logging.Handler):
    """Captures warning messages emitted by the backend solvers to display them in the GUI."""
    def __init__(self):
        super().__init__()
        self.captured_warnings = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.captured_warnings.append(record.getMessage())