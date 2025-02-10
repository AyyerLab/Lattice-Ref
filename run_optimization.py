import h5py
import cupy as cp
import configparser
from cupyx.scipy import ndimage

from utils import init_tobj
from config_functions import run_config
from utils_cupy import do_fft, do_ifft, rotate_arr, shrinkwrap

from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer
from optimize_orient import OrientOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        (self.N, self.NUM_FRAMES, 
         self.NUM_ITER, self.SEED,self.SAMPLE,
         self.INIT_FTOBJ, self.PIXELS, 
         self.USE_SHRINKWRAP,
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

    def symmetrization(self, ftobj):
        angles = [120 * (cp.pi / 180), 240 * (cp.pi / 180)]
        sym_ftobj = ftobj.copy()
        for ang in angles:
            rotated_ftobj = rotate_arr(self.N, ftobj, ang)
            sym_ftobj += rotated_ftobj
        return sym_ftobj

    def run_optimization(self, NUM_ITER=None):
        self.ftobj = self.get_ftobj()
        cp.random.seed(self.SEED+10)
        self.angles = cp.random.uniform(0, 1, size=self.NUM_FRAMES) * 2. * cp.pi

        for i in range(1, self.NUM_ITER + 1):
            parameters_optimizer = ParamOptimizer(i, self.N, self.ftobj, self.data_file, self.output_file, self.angles)
            dx, dy, fluence, _, _ = parameters_optimizer.optimize_params()
            self.shifts = cp.vstack((dx, dy)).T
            self.fluence = fluence

            orientation_optimizer = OrientOptimizer(self.N, self.fluence, self.shifts, self.ftobj, self.data_file)
            angs, _ = orientation_optimizer.optimize_orientation()
            self.angles = angs

            object_optimizer = ObjectOptimizer(i, self.N, self.shifts, self.fluence, self.angles, self.data_file, self.output_file)
            ftobj_curr = object_optimizer.solve()

            apply_shrinkwrap = (self.USE_SHRINKWRAP and i % 5 == 0)
            if apply_shrinkwrap:
                ftobj_curr = shrinkwrap(self.N, ftobj_curr, self.PIXELS, 1.25)


            denominator = cp.where(cp.abs(self.ftobj) == 0, 1e-10, cp.abs(self.ftobj))
            error_ftobj = cp.sum(cp.abs(self.ftobj - ftobj_curr) / denominator).get()

            if self.SAMPLE == 'PS1_10A':
                self.ftobj = self.symmetrization(ftobj_curr)
            else :
                self.ftobj = ftobj_curr

            self.save_results(i, self.shifts, self.fluence, self.angles, self.ftobj, error_ftobj)

        print("Optimization completed.")

    def save_results(self, itern, shifts, fluence, angles, ftobj, error_ftobj):
        output_file = self.output_file.replace('.h5', f'{itern:03}.h5')
        datasets = {
            'fitted_shifts': cp.asnumpy(shifts),
            'fitted_fluence': cp.asnumpy(fluence),
            'fitted_angles': cp.asnumpy(angles),
            'fitted_ftobj': cp.asnumpy(ftobj),
            'error_ftobj': cp.asnumpy(error_ftobj),
        }
        with h5py.File(output_file, 'w') as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    device_id = 1
    cp.cuda.Device(device_id).use()
    runner = OptimizationRunner('config.ini')
    runner.run_optimization()

