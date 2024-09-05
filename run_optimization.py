import numpy as np
import h5py
import sys
import os
from scipy import ndimage

from utils import optim_config
from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer


class OptimizationRunner:
    def __init__(self, config_file):
        (self.N, self.NUM_SAMPLES, self.SCALE, self.SEED, 
         self.INIT_FTOBJ_TYPE, self.DATA_FILE, self.OUTPUT_FILE, self.PIXELS) = optim_config(config_file)

        self.ftobj = None

    def init_ftobj(self, npixs, size):
        obj = np.zeros(size)
        cen = (size[0] // 2, size[1] // 2)
        y, x = np.ogrid[:size[0], :size[1]]
        dcen = np.sqrt((x - cen[1]) ** 2 + (y - cen[0]) ** 2)
        flat_dcen = dcen.flatten()
        sorted_indices = np.argsort(flat_dcen)
        selected_indices = sorted_indices[:npixs]
        obj.flat[selected_indices] = 1
        return np.fft.fftshift(np.fft.fftn(obj))

    def get_ftobj(self):
        # Random Initialization
        if self.INIT_FTOBJ_TYPE == 'RD':
            self.ftobj = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
        # True Solution
        elif self.INIT_FTOBJ_TYPE == 'TS':  # True Solution
            self.ftobj = self.load_ftobj()
        # A Circular Object
        elif self.INIT_FTOBJ_TYPE == 'CR':
            self.ftobj = self.init_ftobj(self.PIXELS, size=(self.N, self.N))
        else:
            raise ValueError(f"Unknown INIT_FTOBJ_TYPE value: {self.INIT_FTOBJ_TYPE}")

    def load_ftobj(self):
        with h5py.File(self.DATA_FILE, 'r') as file:
            return file['ftobj'][:]

    def load_true_values(self):
        with h5py.File(self.DATA_FILE, 'r') as file:
            dx = file['shifts'][:, 0]
            dy = file['shifts'][:, 1]
            fluence = file['fluence'][:]
        return dx, dy, fluence

    def shrinkwrap(self, ftobj_pred, pixels, sig):
        invsuppmask = np.ones((self.N,)*2, dtype=np.bool_)
        amodel = np.abs(np.fft.ifftn(np.fft.ifftshift(ftobj_pred.reshape((self.N,)*2))))
        famodel = ndimage.gaussian_filter(amodel, sig)
        thresh = np.sort(famodel.ravel())[amodel.size - pixels]
        invsuppmask = famodel < thresh
        amodel[invsuppmask] = 0
        ftobj_pred = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(amodel)))
        return ftobj_pred


    def run_optimization(self, TOLERANCE=1e-6, MAX_ITERATIONS=1000, USE_TRUE_VALUES=False):
        self.get_ftobj()

        curr_error = float('inf')
        itern = 1

        # Load true values or initialize Shifts/Fluence
        if USE_TRUE_VALUES:
            dx, dy, fluence = self.load_true_values()
            shifts = np.vstack((dx, dy)).T
        else:
            shifts = None
            fluence = None

        while curr_error > TOLERANCE and itern < MAX_ITERATIONS:
            # Optimize Parameters if not using true values
            if not USE_TRUE_VALUES:
                optimizer = ParamOptimizer(self.N, self.DATA_FILE, self.ftobj, itern)
                fitted_dx, fitted_dy, fitted_fluence, min_error = optimizer.optimize_params()
                shifts = np.vstack((fitted_dx, fitted_dy)).T
                fluence = fitted_fluence
            
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            grid_optimizer = ObjectOptimizer(self.N, self.DATA_FILE, shifts, fluence, self.OUTPUT_FILE, itern)
            optimized_ftobj = grid_optimizer.optimize_all_pixels()

            
            ftobj_curr = optimized_ftobj[:, :, 0] + 1j * optimized_ftobj[:, :, 1]
            if itern % 1 == 0:
                if itern<50:
                    sig=5
                else:
                    sig=2
                ftobj_curr = self.shrinkwrap(ftobj_curr, self.PIXELS, sig)
           
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

            error = np.abs(np.abs(self.ftobj) - np.abs(ftobj_curr)) / np.abs(self.ftobj)
            non_converged_pixels = np.where(error.ravel() > TOLERANCE)[0]


            print(f"Iteration {itern}: Error(FTOBJ) = {error.sum()}")
            print(f"Iteration {itern}: Non-converged pixels = {len(non_converged_pixels)}")

            self.ftobj = ftobj_curr
            curr_error = error.sum()
            itern += 1

        print("Optimization converged.")

if __name__ == "__main__":
    CONFIG_FILE = 'config.ini'
    runner = OptimizationRunner(CONFIG_FILE)
    USE_TRUE_VALUES = False
    runner.run_optimization(USE_TRUE_VALUES=USE_TRUE_VALUES)

