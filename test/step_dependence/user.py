import numpy as np

from step_dependence import *

params = Params(length_x=10, length_y=10, grid_step=0.1)
mass = 1

N = 1000
dt_space = np.linspace(1e-7, 1e-4, N)

modes = [(1, 1)]
coeffs = [1]

ts = TimeStepper()

ts.calc_errors(dt_space, params, modes, coeffs)

s = Saver()

s.from_helper(ts)

s.write('./data/second.pickle')