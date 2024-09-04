import numpy as np
import h5py
import sys
import os

from utils import optim_config
from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer

#from conj_grad import ConjugateGradientOptimizer

def _init_tobj(rad, size=(127, 127)):
    obj = np.zeros(size)
    cen = (size[0] // 2, size[1] // 2)
    y, x = np.ogrid[:size[0], :size[1]]
    dcen = np.sqrt((x - cen[1])**2 + (y - cen[0])**2)
    cmask = dcen <= rad
    obj[cmask] = 1
    return obj, np.fft.fftshift(np.fft.fftn(obj))

class OptimizationRunner:
    def __init__(self, config_file):
        self.N, self.NUM_SAMPLES, self.SCALE, self.SEED, self.INIT_FTOBJ, self.DATA_FILE, self.OUTPUT_FILE = optim_config(config_file)
        self.ftobj = None

    def initialize_ftobj(self):
        if self.INIT_FTOBJ == 'RD':
            self.ftobj = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
        elif self.INIT_FTOBJ == 'TS':
            self.ftobj = self.load_ftobj()
        elif self.INIT_FTOBJ.startswith('CIRCLE'):
            radius = int(self.INIT_FTOBJ.split('_')[1])
            circle, ft_circle = _init_tobj(radius, size=(self.N, self.N))
            self.ftobj = ft_circle
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
                optimizer = ParamOptimizer(self.N, self.DATA_FILE, self.ftobj, INIT_ITER)
                fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
                shifts = np.vstack((fitted_dx, fitted_dy)).T
                fluence = fitted_fluence

            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            grid_optimizer = ObjectOptimizer(self.N, self.DATA_FILE, shifts, fluence, self.OUTPUT_FILE, INIT_ITER)
            optimized_ftobj = grid_optimizer.optimize_all_pixels()
            # cg_optimizer = ConjugateGradientOptimizer(self.N, self.DATA_FILE, self.SCALE, shifts, fluence, self.OUTPUT_FILE, INIT_ITER)
            # optimized_ftobj = cg_optimizer.optimize_all_pixels()

            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            ftobj_curr = optimized_ftobj[:,:,0] + 1j * optimized_ftobj[:,:,1]

            error = np.abs(np.abs(self.ftobj)-np.abs(ftobj_curr))/np.abs(self.ftobj)
            pix_idx = np.where(error.ravel()>TH)[0]
            print(f"ITERATION {INIT_ITER}: ERROR(FTOBJ) = {error.sum()}")
            print(f"ITERATION {INIT_ITER}: LITPIX = {len(pix_idx)}")
            self.ftobj = ftobj_curr
            INIT_ITER += 1
        print("OPTIMIZATION CONVERGED.")

if __name__ == "__main__":
    config_file = 'config.ini'
    runner = OptimizationRunner(config_file)
    use_true_values = True
    runner.run_optimization(use_true_values=use_true_values)

