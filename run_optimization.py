import cupy as cp
import h5py
from cupyx.scipy import ndimage
import configparser

from utils import optim_config
from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.N, _, _, _, self.INIT_FTOBJ_TYPE, self.DATA_FILE, self.OUTPUT_FILE, self.PIXELS = optim_config(config_file)
        self.NUM_ITERATION = int(self.config['OPTIMIZATION']['NUM_ITERATION'])
        self.USE_SHRINKWRAP = self.config['OPTIMIZATION'].getboolean('USE_SHRINKWRAP', fallback=True)
        self.ftobj = None

    def do_fft(self, obj):
        return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(obj)))

    def do_ifft(self, ftobj):
        return cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(ftobj)))

    def init_ftobj(self):
        size = (self.N, self.N)
        obj = cp.zeros(size)
        cen = (self.N // 2, self.N // 2)
        y, x = cp.ogrid[:self.N, :self.N]
        dcen = cp.sqrt((x - cen[1])**2 + (y - cen[0])**2)
        obj.ravel()[cp.argsort(dcen.ravel())[:self.PIXELS]] = 1
        self.ftobj = self.do_fft(obj)

    def get_ftobj(self):
        if self.INIT_FTOBJ_TYPE == 'RD':
            self.ftobj = cp.random.rand(self.N, self.N) + 1j * cp.random.rand(self.N, self.N)
        elif self.INIT_FTOBJ_TYPE == 'TS':
            with h5py.File(self.DATA_FILE, 'r') as file:
                self.ftobj = cp.asarray(file['ftobj'][:])
        elif self.INIT_FTOBJ_TYPE == 'CR':
            self.init_ftobj()
        else:
            raise ValueError(f"Unknown INIT_FTOBJ_TYPE: {self.INIT_FTOBJ_TYPE}")

    def shrinkwrap(self, ftobj_pred, sig):
        invsuppmask = cp.ones((self.N,)*2, dtype=cp.bool_)
        amodel = cp.abs(self.do_ifft(ftobj_pred.reshape((self.N,)*2)))
        famodel = ndimage.gaussian_filter(amodel, sig)
        thresh = cp.sort(famodel.ravel())[amodel.size - self.PIXELS]
        invsuppmask = famodel < thresh
        amodel[invsuppmask] = 0
        return self.do_fft(amodel)

    def run_optimization(self, NUM_ITER=None, USE_TRUE_VALUES=False):
        self.get_ftobj()
        itern = 1

        if NUM_ITER is None:
            NUM_ITER = self.NUM_ITERATION

        if USE_TRUE_VALUES:
            with h5py.File(self.DATA_FILE, 'r') as file:
                shifts = cp.asarray(file['shifts'][:, :2])
                fluence = cp.asarray(file['fluence'][:])
        else:
            shifts = fluence = None

        while itern <= NUM_ITER:
            if not USE_TRUE_VALUES:
                optimizer = ParamOptimizer(self.N, self.DATA_FILE, self.ftobj, itern)
                dx, dy, fluence, _ = optimizer.optimize_params()
                shifts = cp.vstack((dx, dy)).T

            grid_optimizer = ObjectOptimizer(self.N, self.DATA_FILE, shifts, fluence, self.OUTPUT_FILE, itern, self.ftobj)
            optimized_ftobj = grid_optimizer.optimize_all_pixels()
            ftobj_curr = optimized_ftobj
            
            if self.USE_SHRINKWRAP and itern >= 50 and itern % 5 == 0:
                sig = 2
                ftobj_curr = self.shrinkwrap(ftobj_curr, sig)

            denominator = cp.abs(self.ftobj)
            denominator = cp.where(denominator == 0, 1e-10, denominator)
            error = cp.sum(cp.abs(cp.abs(self.ftobj) - cp.abs(ftobj_curr)) / denominator).get()

            print(f"Iteration {itern}: Error = {error}")

            self.ftobj = ftobj_curr
            itern += 1

        print("Optimization completed.")

if __name__ == "__main__":
    runner = OptimizationRunner('config.ini')
    runner.run_optimization(USE_TRUE_VALUES=False)

