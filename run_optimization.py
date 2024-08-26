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

    def load_true_values(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            fluence = f['fluence'][:]
            dx = f['shifts'][:,0]
            dy = f['shifts'][:,1]
        return dx, dy, fluence

    def run_optimization(self, TH=1e-6, M_ITER=1000, use_true_values=False):
        self.initialize_ftobj()
        err = float('inf')
        INIT_ITER = 1

        if use_true_values:
            dx, dy, fluence = self.load_true_values()
            shifts = np.vstack((dx, dy)).T
        else:
            shifts = None
            fluence = None

        while err > TH and INIT_ITER < M_ITER:
            if not use_true_values:
                optimizer = Optimizer(self.N, self.DATA_FILE, self.SCALE, self.ftobj, INIT_ITER)
                fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
                shifts = np.vstack((fitted_dx, fitted_dy)).T
                fluence = fitted_fluence

            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            cg_optimizer = ConjugateGradientOptimizer(self.N, self.DATA_FILE, self.SCALE, shifts, fluence, self.OUTPUT_FILE, INIT_ITER)
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
    use_true_values = False 
    runner.run_optimization(use_true_values=use_true_values)

