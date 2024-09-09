import cupy as cp
import numpy as np
import h5py
import sys
import os
from cupyx.scipy import ndimage

from utils import optim_config
from optimize_params import ParamOptimizer  # Assuming this is the GPU-optimized version
from optimize_ftobj import ObjectOptimizer  # Assuming this is the GPU-optimized version

class OptimizationRunner:
    def __init__(self, config_file):
        (self.N, self.NUM_SAMPLES, self.SCALE, self.SEED, 
         self.INIT_FTOBJ_TYPE, self.DATA_FILE, self.OUTPUT_FILE, self.PIXELS) = optim_config(config_file)

        self.ftobj = None

    def init_ftobj(self, npixs, size):
        obj = cp.zeros(size)
        cen = (size[0] // 2, size[1] // 2)
        y, x = cp.ogrid[:size[0], :size[1]]
        dcen = cp.sqrt((x - cen[1]) ** 2 + (y - cen[0]) ** 2)
        flat_dcen = dcen.flatten()
        sorted_indices = cp.argsort(flat_dcen)
        selected_indices = sorted_indices[:npixs]
    
        # Use ravel() to create a flat array that supports advanced indexing
        obj_flat = obj.ravel()
        obj_flat[selected_indices] = 1
        obj = obj_flat.reshape(size)

        return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(obj)))

    def get_ftobj(self):
        if self.INIT_FTOBJ_TYPE == 'RD':
            self.ftobj = cp.random.rand(self.N, self.N) + 1j * cp.random.rand(self.N, self.N)
        elif self.INIT_FTOBJ_TYPE == 'TS':
            self.ftobj = cp.asarray(self.load_ftobj())
        elif self.INIT_FTOBJ_TYPE == 'CR':
            self.ftobj = self.init_ftobj(self.PIXELS, size=(self.N, self.N))
        else:
            raise ValueError(f"Unknown INIT_FTOBJ_TYPE value: {self.INIT_FTOBJ_TYPE}")

    def load_ftobj(self):
        with h5py.File(self.DATA_FILE, 'r') as file:
            return file['ftobj'][:]

    def load_true_values(self):
        with h5py.File(self.DATA_FILE, 'r') as file:
            dx = cp.asarray(file['shifts'][:, 0])
            dy = cp.asarray(file['shifts'][:, 1])
            fluence = cp.asarray(file['fluence'][:])
        return dx, dy, fluence

    def shrinkwrap(self, ftobj_pred, pixels, sig):
        invsuppmask = cp.ones((self.N,)*2, dtype=cp.bool_)
        amodel = cp.abs(cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(ftobj_pred.reshape((self.N,)*2))))))
        famodel = ndimage.gaussian_filter(amodel, sig)
        thresh = cp.sort(famodel.ravel())[amodel.size - pixels]
        invsuppmask = famodel < thresh
        amodel[invsuppmask] = 0
        ftobj_pred = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(amodel)))
        return ftobj_pred

    def run_optimization(self, TOLERANCE=1e-6, MAX_ITERATIONS=1000, USE_TRUE_VALUES=False):
        self.get_ftobj()

        curr_error = float('inf')
        itern = 1

        if USE_TRUE_VALUES:
            dx, dy, fluence = self.load_true_values()
            shifts = cp.vstack((dx, dy)).T
        else:
            shifts = None
            fluence = None

        while curr_error > TOLERANCE and itern < MAX_ITERATIONS:
            if not USE_TRUE_VALUES:
                optimizer = ParamOptimizer(self.N, self.DATA_FILE, self.ftobj, itern)
                fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
                shifts = cp.vstack((fitted_dx, fitted_dy)).T
                fluence = fitted_fluence
            
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            grid_optimizer = ObjectOptimizer(self.N, self.DATA_FILE, shifts, fluence, self.OUTPUT_FILE, itern)
            optimized_ftobj = grid_optimizer.optimize_all_pixels()

            ftobj_curr = optimized_ftobj[:, :, 0] + 1j * optimized_ftobj[:, :, 1]
            if itern % 10 == 0:
                sig = 5 if itern < 50 else 2
                ftobj_curr = self.shrinkwrap(ftobj_curr, self.PIXELS, sig)
           
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            error = cp.abs(cp.abs(self.ftobj) - cp.abs(ftobj_curr)) / cp.abs(self.ftobj)
            non_converged_pixels = cp.where(error.ravel() > TOLERANCE)[0]

            print(f"Iteration {itern}: Error(FTOBJ) = {error.sum().get()}")
            print(f"Iteration {itern}: Non-converged pixels = {len(non_converged_pixels)}")

            self.ftobj = ftobj_curr
            curr_error = error.sum().get()
            itern += 1

        print("Optimization converged.")

if __name__ == "__main__":
    CONFIG_FILE = 'config.ini'
    runner = OptimizationRunner(CONFIG_FILE)
    USE_TRUE_VALUES = False
    runner.run_optimization(USE_TRUE_VALUES=USE_TRUE_VALUES)
