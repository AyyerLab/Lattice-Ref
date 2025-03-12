import h5py
import cupy as cp
from cupyx.scipy import ndimage
from utils import init_tobj
from config_functions import run_config
from utils_cupy import do_fft, do_ifft, rotate_arr, shrinkwrap
from optimize_params import ParamOptimizer
from optimize_ftobj import ObjectOptimizer
from optimize_orient import OrientOptimizer

class OptimizationRunner:
    def __init__(self, config_file):
        self.N, self.NUM_FRAMES, self.NUM_ITER, self.SEED, self.SAMPLE, self.INIT_FTOBJ, self.PIXELS, \
        self.USE_SHRINKWRAP, self.data_file, self.output_file = run_config(config_file)

        self.ftobj = self.get_ftobj()
        cp.random.seed(self.SEED + 10)
        self.angles = cp.random.uniform(0, 2 * cp.pi, size=self.NUM_FRAMES)
        self.shifts = None
        self.fluence = None

    def get_ftobj(self):
        fobjs = {
            'RD': lambda: cp.random.rand(self.N, self.N) + 1j * cp.random.rand(self.N, self.N),
            'TS': lambda: cp.asarray(h5py.File(self.data_file, 'r')['ftobj'][:]),
            'CR': lambda: do_fft(init_tobj(self.N, self.PIXELS))
        }
        return fobjs[self.INIT_FTOBJ]()

    def run_optimization(self, num_iter=None):
        num_iter = num_iter or self.NUM_ITER
        for i in range(1, num_iter + 1):
            params_optimizer = ParamOptimizer(i, self.N, self.ftobj, self.data_file, self.angles)
            dx, dy, fluence, _, _ = params_optimizer.optimize_params()
            self.shifts = cp.vstack((dx, dy)).T
            self.fluence = fluence

            orient_optimizer = OrientOptimizer(self.N, self.fluence, self.shifts, self.ftobj, self.data_file)
            self.angles, _ = orient_optimizer.optimize_orientation()

            obj_optimizer = ObjectOptimizer(i, self.N, self.shifts, self.fluence, self.angles, self.data_file, self.output_file)
            ftobj_curr = obj_optimizer.solve()

            if self.USE_SHRINKWRAP and i % 2 == 0:
                ftobj_curr = shrinkwrap(self.N, ftobj_curr, self.PIXELS, 1.25)

            self.ftobj = ftobj_curr
            self.save_results(i, self.shifts, self.fluence, self.angles, self.ftobj)

        print("Optimization completed.")

    def save_results(self, iteration, shifts, fluence, angles, ftobj):
        output_file = self.output_file.replace('.h5', f'{iteration:03}.h5')
        datasets = {
            'fitted_shifts': cp.asnumpy(shifts),
            'fitted_fluence': cp.asnumpy(fluence),
            'fitted_angles': cp.asnumpy(angles),
            'fitted_ftobj': cp.asnumpy(ftobj)
        }
        with h5py.File(output_file, 'w') as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    #device_id = 1
    #cp.cuda.Device(device_id).use()
    runner = OptimizationRunner('config.ini')
    runner.run_optimization()
