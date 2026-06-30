import unittest
import numpy as np
from step_dependence import Params, TimeStepper, plotting


class TestSolversFirstStep(unittest.TestCase):
    def setUp(self):
        self.params = Params(length_x=16, length_y=16, grid_step=0.1)
        self.N = 20
        self.dt_space = np.logspace(-7, -4, self.N)
        self.modes = [(1, 1)]
        self.coeffs = [1]

        self.ts = TimeStepper()
        self.ts.calc_errors(self.dt_space, self.params, self.modes, self.coeffs)
        self.norm_names = ["sup", "l2"]
        # fig, ax = plotting(self.ts, show=True)

    def assert_errors_valid(self, solver_names):
        for solvername in solver_names:
            for normname in self.norm_names:
                with self.subTest(solver=solvername, norm=normname):
                    prev_error = -1.0

                    for k in self.ts.results[(solvername, normname)]:
                        self.assertLess(k, 0.01)

                        # smaller \delta t must result in smaller difference
                        # difference should converge to 0 with \delta t -> 0
                        if prev_error != -1.0:
                            self.assertGreater(k, prev_error)

                        prev_error = k

    def test_cn(self):
        self.assert_errors_valid(["cn"])

    def test_ssfm(self):
        self.assert_errors_valid(["ssfm", "sym_ssfm"])


if __name__ == "__main__":
    unittest.main()
