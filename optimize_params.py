import numpy as np
import h5py
import sys
from scipy.optimize import minimize
from utils import get_vals

class Optimizer:
    def __init__(self, N, DATA_FILE, SCALE, ftobj, INIT_ITER):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.SCALE = SCALE
        self.INIT_ITER = INIT_ITER
        self.ftobj = ftobj
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.funitc = f['funitc'][:]
            self.fluence_vals = f['fluence'][:]

    def analyze_frame(self, intens, hk):
        funitc_vals = np.array([get_vals(self.funitc, self.cen, *val) for val in hk])
        ftobj_vals = np.array([get_vals(self.ftobj, self.cen, *val) for val in hk])
        intens_vals = np.array([get_vals(intens, self.cen, *val) for val in hk])

        def objective(params):
            dx, dy, fluence = params
            qh = np.array([h for h, k in hk])
            qk = np.array([k for h, k in hk])
            phase = 2.0 * np.pi * (qh * dx + qk * dy)
            pramp = np.exp(1j * phase)
            model_int = fluence * np.abs(self.SCALE * funitc_vals + ftobj_vals * pramp) ** 2
            error = np.sum((model_int - intens_vals) ** 2)
            return error

        dx_range = np.linspace(0, 1, 4)
        dy_range = np.linspace(0, 1, 4)
        fluence_range = np.arange(0, 10, 0.5)
        dx_grid, dy_grid, fluence_grid = np.meshgrid(dx_range, dy_range, fluence_range)

        params_grid = np.array([dx_grid.ravel(), dy_grid.ravel(), fluence_grid.ravel()]).T
        min_error = np.finfo('f8').max
        optimal_params = None

        for params in params_grid:
            res = minimize(objective, params, method='L-BFGS-B')
            if res.fun < min_error:
                min_error = res.fun
                optimal_params = res.x

        return optimal_params, min_error

    def _optimize_frame(self, frame_idx, hk):
        intens = self.intens_vals[frame_idx]
        optimal_params, min_error = self.analyze_frame(intens, hk)
        result = {
            'frame_idx': frame_idx,
            'fitted_dx': optimal_params[0] % 1,
            'fitted_dy': optimal_params[1] % 1,
            'fitted_fluence': optimal_params[2] % 1,
            'min_error': min_error
        }
        return result


    def optimize_params(self):
        hk = [(0, 1), (1, 1), (1, 0), (1, -1)]
        frames = range(self.intens_vals.shape[0])

        fitted_dx = []
        fitted_dy = []
        fitted_fluence = []
        min_error = []

        for frame_idx in frames:
            result = self._optimize_frame(frame_idx, hk)
            fitted_dx.append(result['fitted_dx'])
            fitted_dy.append(result['fitted_dy'])
            fitted_fluence.append(result['fitted_fluence'])
            min_error.append(result['min_error'])
            print((
                f"\rITER {self.INIT_ITER}: "
                f"FRAME {frame_idx}/{len(frames)}: "
                f"Dx={result['fitted_dx']:.3f}, "
                f"Dy={result['fitted_dy']:.3f}, "
                f"FLUENCE={result['fitted_fluence']:.3f}, "
                f"ERROR={result['min_error']:.3e}"
            ), end='', file=sys.stdout)
            sys.stdout.flush()

        return np.array(fitted_dx), np.array(fitted_dy), np.array(fitted_fluence), np.array(min_error)
