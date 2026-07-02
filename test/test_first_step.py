"""
Calculate maximal deviation between simulation and analytical solution after one step
for different values of time step: delta t.
Test is passed when the deviation stays below a threshold:
for 1e-4 : 1%
for 1e-5 : 0.1%
for 1e-6 : 0.01%
for 1e-7 : 0.001%
"""

import numpy as np
import pytest
from step_dependence.step_dependence import Params, TimeStepper


class TestSolversFirstStep:

    @pytest.fixture(autouse=True)
    def setup_stepper(self):
        self.params = Params(length_x=16, length_y=16, grid_step=0.1)
        self.N = 20
        self.dt_space = [1e-7, 1e-6, 1e-5, 1e-4]
        self.thresholds = [1e-4, 1e-3, 1e-2, 1e-2]
        self.modes = [(1, 1)]
        self.coeffs = [1]

        self.ts = TimeStepper()
        self.ts.calc_errors(self.dt_space, self.params, self.modes, self.coeffs)
        self.norm_names = ["sup", "l2"]

    def _assert_errors_valid(self, solver_names, subtests):
        for solvername in solver_names:
            for normname in self.norm_names:
                with subtests.test(
                    msg=f"{solvername}-{normname}", solver=solvername, norm=normname
                ):
                    for res, threshold in zip(self.ts.results[(solvername, normname)], self.thresholds):
                        assert res < threshold, f"Difference is bigger than {threshold*100}%"

    def test_cn(self, subtests):
        self._assert_errors_valid(["cn"], subtests)

    def test_ssfm(self, subtests):
        self._assert_errors_valid(["ssfm", "sym_ssfm"], subtests)
