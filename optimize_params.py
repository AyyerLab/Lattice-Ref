import numpy as np
from scipy.optimize import minimize
import h5py
import configparser

class Optimizer:
    def __init__(self, N, DATA_FILE, SCALE, OUTPUT_FILE):
        self.N = N
        self.cen = self.N // 2
        self.DATA_FILE = DATA_FILE
        self.SCALE = SCALE
        self.OUTPUT_FILE = OUTPUT_FILE
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.ftobj = f['ftobj'][:]
            self.funitc = f['funitc'][:]
            self.shifts = f['shifts'][:]
            self.fluence_vals = f['fluence'][:]

    def _getvals(self, array, h, k):
        qh = h + self.cen
        qk = k + self.cen
        return array[qh, qk]

    def analyze_frame(self, intens, hk):
        funitc_vals = [self._getvals(self.funitc, *val) for val in hk]
        ftobj_vals = [self._getvals(self.ftobj, *val) for val in hk]
        intens_vals = [self._getvals(intens, *val) for val in hk]

        def objective(params):
            dx, dy, fluence = params
            error = 0
            for (h, k), funitc_val, ftobj_val, intens_val in zip(hk, funitc_vals, ftobj_vals, intens_vals):
                qh = h
                qk = k
                phase = 2.0 * np.pi * (qh * dx + qk * dy)
                pramp = np.exp(1j * phase)
                model_int = fluence * np.abs(self.SCALE * funitc_val + ftobj_val * pramp) ** 2
                error += (model_int - intens_val) ** 2
            return error

        ncoarse = 4
        fluence_range = np.arange(0, 10, 0.5)
        dx_range = np.arange(0, 1, 1 / ncoarse)
        dy_range = np.arange(0, 1, 1 / ncoarse)

        min_error = np.finfo('f8').max
        optimal_params = None

        for dx in dx_range:
            for dy in dy_range:
                for fl in fluence_range:
                    res = minimize(objective, (dx, dy, fl), method='Nelder-Mead')
                    if res.fun < min_error:
                        min_error = res.fun
                        optimal_params = res.x

        return optimal_params, min_error

    def optimize_params(self):
        hk = [(0, 1), (1, 1), (1, 0), (1, -1)]
        results = []

        for frame_idx in range(self.intens_vals.shape[0]):
            intens = self.intens_vals[frame_idx]
            optimal_params, min_error = self.analyze_frame(intens, hk)

            result = {
                'frame_idx': frame_idx,
                'fitted_dx': optimal_params[0] % 1,
                'fitted_dy': optimal_params[1] % 1,
                'fitted_fluence': optimal_params[2] % 1,
                'min_error': min_error
            }

            print(f"Frame {frame_idx}:")
            print(f"Fitted: dx = {result['fitted_dx']}, dy = {result['fitted_dy']}, fl = {result['fitted_fluence']}")
            print(f"Error: {min_error}\n")

            results.append(result)

        self.save_results(results)

    def save_results(self, results):
        with h5py.File(self.OUTPUT_FILE, 'w') as f:
            fitted_dx = [result['fitted_dx'] for result in results]
            fitted_dy = [result['fitted_dy'] for result in results]
            fitted_fluence = [result['fitted_fluence'] for result in results]
            min_error = [result['min_error'] for result in results]

            f['fitted_dx'] = fitted_dx
            f['fitted_dy'] = fitted_dy
            f['fitted_fl'] = fitted_fluence
            f['error'] = min_error

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    N = config.getint('DATA_GENERATION', 'N')
    SCALE = config.getint('DATA_GENERATION', 'SCALE')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')
    OUTPUT_FILE = config.get('OPTIMIZATION', 'OUTPUT_FILE')
    return N, DATA_FILE, SCALE, OUTPUT_FILE

if __name__ == "__main__":
    config_file = 'config.ini'
    N, DATA_FILE, SCALE, OUTPUT_FILE = load_config(config_file)
    optimizer = Optimizer(N, DATA_FILE, SCALE, OUTPUT_FILE)
    optimizer.optimize_params()

