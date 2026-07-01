"""
Calculate maximal difference between simulation and analytical solution after one step for different values of time step: delta t.
Test is passed when difference in a given range is lower than 1% and there is a positive correlation between difference and delta t.
"""

import numpy as np
import pytest
from step_dependence.step_dependence import Params, TimeStepper
from scipy.stats import kendalltau


class TestSolversFirstStep:

    @pytest.fixture(autouse=True)
    def setup_stepper(self):
        self.params = Params(length_x=16, length_y=16, grid_step=0.1)
        self.N = 20
        self.dt_space = np.logspace(-7, -4, self.N)
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

                    for k in self.ts.results[(solvername, normname)]:
                        assert k < 0.01, f"Difference is bigger than 1%"

                    tau, _ = kendalltau(
                        self.dt_space, self.ts.results[(solvername, normname)]
                    )
                    assert (
                        tau > 0.95
                    ), f"The difference should decrease as dt decreases, correlation is too low {tau}"

    def test_cn(self, subtests):
        self._assert_errors_valid(["cn"], subtests)

    def test_ssfm(self, subtests):
        self._assert_errors_valid(["ssfm", "sym_ssfm"], subtests)
