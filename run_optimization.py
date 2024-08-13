import numpy as np
import h5py
import sys
import os

from utils import optim_config
from optimize_params import Optimizer
from conj_grad import ConjugateGradientOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        self.N, self.NUM_SAMPLES, self.SCALE, self.SEED, self.INIT_FTOBJ, self.DATA_FILE, self.OUTPUT_FILE = optim_config(config_file)
        self.ftobj = None

    def initialize_ftobj(self):
        if self.INIT_FTOBJ == 'RD':
            self.ftobj = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
        elif self.INIT_FTOBJ == 'TS':
            self.ftobj = self.load_ftobj()
        else:
            raise ValueError(f"Unknown INIT_FTOBJ value: {self.INIT_FTOBJ}")

    def load_ftobj(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            return f['ftobj'][:]

    def run_optimization(self, TH=1e-6, M_ITER=500):
        self.initialize_ftobj()
        err = float('inf')
        INIT_ITER = 1
        while err > TH and INIT_ITER < M_ITER:

            optimizer = Optimizer(self.N, self.DATA_FILE, self.SCALE, self.ftobj, INIT_ITER)
            fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
            shifts = np.vstack((fitted_dx, fitted_dy)).T
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            cg_optimizer = ConjugateGradientOptimizer(self.N, self.DATA_FILE, self.SCALE, shifts, fitted_fluence, self.OUTPUT_FILE)
            optimized_ftobj = cg_optimizer.optimize_all_pixels()

            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            ftobj_curr = optimized_ftobj[:,:,0] + 1j * optimized_ftobj[:,:,1]
            err = np.linalg.norm(ftobj_curr - self.ftobj)
            print(f"ITERATION {INIT_ITER}: ERROR(FTOBJ) = {err}")
            self.ftobj = ftobj_curr
            INIT_ITER += 1


        print("OPTIMIZATION CONVERGED.")

if __name__ == "__main__":
    config_file = 'config.ini'
    runner = OptimizationRunner(config_file)
    runner.run_optimization()
