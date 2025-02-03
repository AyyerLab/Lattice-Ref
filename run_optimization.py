import cupy as cp
from cupyx.scipy import ndimage
import h5py
import configparser

from config_functions import run_config
from utils import init_tobj
from utils_cupy import do_fft, do_ifft

from optimize_params import ParamOptimizer
from optimize_orient import OrientOptimizer
from optimize_ftobj import ObjectOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        (self.N, self.NUM_FRAMES, 
         self.NUM_ITER, self.SEED,
         self.INIT_FTOBJ, self.PIXELS, 
         self.USE_SHRINKWRAP, self.APPLY_ORIENTATION,
         self.data_file, self.output_file) = run_config(config_file)

        self.ftobj = None
        self.shifts = None
        self.fluence = None
        self.angles = None

    def get_ftobj(self):
        fobjs = {'RD': lambda: cp.random.rand(self.N, self.N) + 1j * cp.random.rand(self.N, self.N),
                 'TS': lambda: cp.asarray(h5py.File(self.data_file, 'r')['ftobj'][:]),
                 'CR': lambda: do_fft(init_tobj(self.N, self.PIXELS))}
        return fobjs[self.INIT_FTOBJ]()

    def shrinkwrap(self, ftobj, sig):
        invsuppmask = cp.ones((self.N,) * 2, dtype=cp.bool_)
        amodel = cp.real(do_ifft(ftobj.reshape((self.N,) * 2)))
        samodel = ndimage.gaussian_filter(amodel, sig)
        thresh = cp.quantile(samodel, (samodel.size - self.PIXELS) / samodel.size)
        invsuppmask = samodel < thresh
        amodel[invsuppmask] = 0
        return do_fft(amodel)

    def run_optimization(self, NUM_ITER=None):
        self.ftobj = self.get_ftobj()
        cp.random.seed(self.SEED+10)
        self.angles = cp.random.uniform(0, 1, size=self.NUM_FRAMES) * 2. * cp.pi

        for i in range(1, self.NUM_ITER + 1):
            # Shifts and Fluence Optimization
            optimizerP = ParamOptimizer(i, self.N, self.ftobj, self.data_file, self.output_file, self.APPLY_ORIENTATION, self.angles)
            dx, dy, fluence, error_p, iter_p = optimizerP.optimize_params()
            self.shifts = cp.vstack((dx, dy)).T
            self.fluence = fluence

            if self.APPLY_ORIENTATION:
                # Orientation Optimization
                optimizerO = OrientOptimizer(self.N, self.fluence, self.shifts, self.ftobj, self.data_file)
                angs, iter_o = optimizerO.optimize_orientation()
            else:
                angs = cp.zeros(self.NUM_FRAMES)
                iter_o = 0
            self.angles = angs

            # Ftobj Optimization
            optimizerOBJ = ObjectOptimizer(i, self.N, self.shifts, self.fluence, self.angles, self.data_file, self.output_file)
            ftobj_ = optimizerOBJ.solve()

            # Apply shrinkwrap
            apply_shrinkwrap = (self.USE_SHRINKWRAP and i == self.NUM_ITER)
            if apply_shrinkwrap:
                ftobj_ = self.shrinkwrap(ftobj_, 1.25)

            # Calculate error
            denominator = cp.where(cp.abs(self.ftobj) == 0, 1e-10, cp.abs(self.ftobj))
            error_ftobj = cp.sum(cp.abs(self.ftobj - ftobj_) / denominator).get()

            print(f"Iteration {i}: Error = {error_ftobj}")
            self.ftobj = ftobj_

            # Save results
            self.save_results(i, self.shifts, self.fluence, self.angles, self.ftobj, error_p, iter_p, iter_o, error_ftobj)

        print("Optimization completed.")

    def save_results(self, itern, shifts, fluence, angles, ftobj, error_p, iter_p, iter_o, error_ftobj):
        output_file = self.output_file.replace('.h5', f'{itern:03}.h5')
        datasets = {
            'fitted_shifts': cp.asnumpy(shifts),
            'fitted_fluence': cp.asnumpy(fluence),
            'fitted_angles': cp.asnumpy(angles),
            'fitted_ftobj': cp.asnumpy(ftobj),
            'error_params': cp.asnumpy(error_p),
            'iter_params': cp.asnumpy(iter_p),
            'iter_orient': cp.asnumpy(iter_o),
            'error_ftobj': cp.asnumpy(error_ftobj),
        }
        with h5py.File(output_file, 'w') as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
#    device_id = 1
#    cp.cuda.Device(device_id).use()
    runner = OptimizationRunner('config.ini')
    runner.run_optimization()

