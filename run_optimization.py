import numpy as np
import h5py
import sys

from utils import load_config
from optimize_params import Optimizer
from conj_grad import ConjugateGradientOptimizer


#Random Initialization
np.random.seed(42)
ftobj = np.random.rand(127, 127) + 1j * np.random.rand(127, 127)

def run_optimization(N, DATA_FILE, SCALE, ftobj, OUTPUT_FILE, th=1e-6, m_iter=100):
    err = float('inf')
    init_iter = 0
    while err > th and init_iter < m_iter:
        optimizer = Optimizer(N, DATA_FILE, SCALE, ftobj)
        fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
        shifts = np.vstack((fitted_dx, fitted_dy)).T

        cg_optimizer = ConjugateGradientOptimizer(N, DATA_FILE, SCALE, shifts, fitted_fluence, OUTPUT_FILE)
        optimized_ftobj = cg_optimizer.optimize_all_pixels()

        ftobj_curr = optimized_ftobj[:,:,0] + 1j* optimized_ftobj[:,:,1]
        err = np.linalg.norm(ftobj_curr - ftobj)
        ftobj = ftobj_curr
        init_iter += 1

        print(f"Iteration {init_iter}, Error: {err}")
    print("Optimization converged.")


#RUN OPTIMIZATION
config_file = 'config.ini'
N, DATA_FILE, SCALE, OUTPUT_FILE = load_config(config_file)
run_optimization(N, DATA_FILE, SCALE, ftobj, OUTPUT_FILE)

