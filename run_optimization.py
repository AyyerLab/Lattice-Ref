import cupy as cp
import h5py
from cupyx.scipy import ndimage
import configparser

from utils import run_config, init_tobj

from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        (self.N,
         self.NUM_ITER,
         self.INIT_FTOBJ,
         self.DATA_FILE,
         self.OUTPUT_FILE,
         self.PIXELS,
         self.USE_SHRINKWRAP,
         self.SHRINKWRAP_RUNS) = run_config(config_file)
        self.ftobj = None

    def do_fft(self, obj):
        return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(obj)))

    def do_ifft(self, ftobj):
        return cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(ftobj)))

    def get_ftobj(self):
        fobjs = {
                'RD': lambda: cp.random.rand(self.N, self.N) + 1j * cp.random.rand(self.N, self.N),
                'TS': lambda: cp.asarray(h5py.File(self.DATA_FILE, 'r')['ftobj'][:]),
                'CR': lambda: self.do_fft(init_tobj(self.N, self.PIXELS))
                }
        if self.INIT_FTOBJ not in fobjs:
            raise ValueError(f"Unknown INIT_FTOBJ: {self.INIT_FTOBJ}")
        return fobjs[self.INIT_FTOBJ]()

    def shrinkwrap(self, ftobj_pred, sig):
        invsuppmask = cp.ones((self.N,)*2, dtype=cp.bool_)
        amodel = cp.real(self.do_ifft(ftobj_pred.reshape((self.N,)*2)))
        samodel = ndimage.gaussian_filter(amodel, sig)
        thresh = cp.quantile(samodel, (samodel.size - self.PIXELS) / samodel.size)
        invsuppmask = samodel < thresh
        amodel[invsuppmask] = 0
        return self.do_fft(amodel)

    def run_optimization(self, NUM_ITER=None):
        self.ftobj = self.get_ftobj()
        
        for i in range(1, self.NUM_ITER + 1):
            # Shifts and Fluence Optimization
            optimizer = ParamOptimizer(i, self.N, self.ftobj, self.DATA_FILE, self.OUTPUT_FILE)
            dx, dy, fluence, error_params, iter_params = optimizer.optimize_params()
            shifts = cp.vstack((dx, dy)).T

            # Ftobj Optimization
            grid_optimizer = ObjectOptimizer(i, self.N, shifts, fluence, self.ftobj, self.DATA_FILE, self.OUTPUT_FILE)
            ftobj_curr, iter_ftobj = grid_optimizer.optimize_all_pixels()

            # Apply shrinkwrap
            apply_shrinkwrap = (self.USE_SHRINKWRAP and
                                i > self.SHRINKWRAP_RUNS[0] and
                                i % self.SHRINKWRAP_RUNS[1] == 0)

            if apply_shrinkwrap:
                ftobj_curr = self.shrinkwrap(ftobj_curr, 0.5)

            # Calculate error
            denominator = cp.where(cp.abs(self.ftobj) == 0, 1e-10, cp.abs(self.ftobj))
            error_ftobj = cp.sum(cp.abs(cp.abs(self.ftobj) - cp.abs(ftobj_curr)) / denominator).get()

            print(f"Iteration {i}: Error = {error_ftobj}")
            self.ftobj = ftobj_curr

            # Save results
            self.save_results(i, shifts, fluence, self.ftobj, error_params, iter_params, error_ftobj, iter_ftobj)

        print("Optimization completed.")

    def save_results(self, itern, shifts, fluence, ftobj, error_params, iter_params, error_ftobj, iter_ftobj):
        output_file = self.OUTPUT_FILE.replace('.h5', f'{itern:03}.h5')
        datasets = {
            'fitted_dx': cp.asnumpy(shifts[:, 0]),
            'fitted_dy': cp.asnumpy(shifts[:, 1]),
            'fitted_fluence': cp.asnumpy(fluence),
            'fitted_ftobj': cp.asnumpy(ftobj),
            'error_params': cp.asnumpy(error_params),
            'iter_params': cp.asnumpy(iter_params),
            'error_ftobj': cp.asnumpy(error_ftobj),
            'iter_ftobj': cp.asnumpy(iter_ftobj)
        }
        with h5py.File(output_file, 'w') as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    runner = OptimizationRunner('config.ini')
    runner.run_optimization()

