import cupy as cp
import h5py
import sys
from configparser import ConfigParser
from utils import get_vals

class ParamOptimizer:
    def __init__(self, i, N, ftobj, data_file, output_file):
        self.N = N
        self.cen = N // 2

        self.DATA_FILE = data_file
        self.OUTPUT_FILE = output_file
        self.ITER = i

        self.ftobj = cp.asarray(ftobj)
        self.SWITCH_TO_REFINEMENT = 50
        self.load_dataset()

    def load_dataset(self):
        # Load Dataset
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])

    def load_fitvals(self, output_file, itern):
        # Load Fitted Parameter Values from Previous Iteration
        file = f'{output_file.split(".h5")[0]}{itern - 1:03d}.h5'
        with h5py.File(file, "r") as f:
            fitvals = tuple(cp.asarray(f[key][:]) for key in ['fitted_dx', 'fitted_dy', 'fitted_fluence', 'error'])
        return fitvals

    # GRID SEARCH METHOD
    def grid_search(self, qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range):
        dx_grid, dy_grid, fluence_grid = cp.meshgrid(dx_range, dy_range, fluence_range, indexing='ij')
        dx_grid = dx_grid.ravel()
        dy_grid = dy_grid.ravel()
        fluence_grid = fluence_grid.ravel()

        phase_grid = 2 * cp.pi * (qh[:, None] * dx_grid + qk[:, None] * dy_grid)
        pramp_grid = cp.exp(1j * phase_grid)

        model_intensity = cp.abs(funitc_vals[:, None] + fluence_grid * ftobj_vals[:, None] * pramp_grid) ** 2
        error = cp.sum((model_intensity - intens_vals[:, None]) ** 2, axis=0)

        min_idx = cp.argmin(error)
        optimal_params = dx_grid[min_idx], dy_grid[min_idx], fluence_grid[min_idx]
        min_error = error[min_idx]

        return optimal_params, min_error

    #COARSE and FINE GRID SEARCH
    def analyze_frame(self, intens, hk, prev_fitvals=None):
        hk = cp.array(hk)
        qh, qk = hk[:, 0], hk[:, 1]

        funitc_vals = get_vals(self.funitc, self.cen, qh, qk)
        ftobj_vals = get_vals(self.ftobj, self.cen, qh, qk)
        intens_vals = get_vals(intens, self.cen, qh, qk)

        if prev_fitvals:
            dx0, dy0, fluence0, min_error = prev_fitvals
        else:
            ncoarse = 300
            dx_range = cp.linspace(0, 1, ncoarse)
            dy_range = cp.linspace(0, 1, ncoarse)
            fluence_range = cp.linspace(0.1, 10, ncoarse)

            optimal_params, min_error = self.grid_search(
                qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range
            )
            dx0, dy0, fluence0 = optimal_params

        # Refinement loop
        gsize = 0.05
        threshold = 1e-4
        for _ in range(1000):
            dx_range = cp.linspace(max(dx0 - gsize, 0), min(dx0 + gsize, 1), 10)
            dy_range = cp.linspace(max(dy0 - gsize, 0), min(dy0 + gsize, 1), 10)
            fluence_range = cp.linspace(max(fluence0 - 2 * gsize, 0.1), min(fluence0 + 2 * gsize, 10), 10)

            new_params, new_error = self.grid_search(
                qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range
            )

            if cp.abs(min_error - new_error) < threshold:
                break

            dx0, dy0, fluence0 = new_params
            min_error = new_error
            gsize /= 2

        return (dx0, dy0, fluence0), float(min_error)


    def optimize_params(self):
        #RUN OPTIMZATION for EACH FRAME
        hk = [(0, 1), (1, 1), (1, 0), (1, -1)]
        num_frames = self.intens_vals.shape[0]

        prev_fitvals = None
        if self.ITER == self.SWITCH_TO_REFINEMENT:
            prev_fitvals = self.load_fitvals(self.OUTPUT_FILE, self.ITER)

        results = []
        for frame_idx in range(num_frames):
            intens = self.intens_vals[frame_idx]
            if prev_fitvals:
                prev_vals_frame = (prev_fitvals[0][frame_idx], prev_fitvals[1][frame_idx], prev_fitvals[2][frame_idx], prev_fitvals[3][frame_idx])
            else:
                prev_vals_frame = None

            optimal_params, error = self.analyze_frame(intens, hk, prev_vals_frame)

            dx, dy, fluence = optimal_params
            results.append({
                'dx': dx % 1,
                'dy': dy % 1,
                'fluence': fluence,
                'error': error
            })

            print(
                f"ITER {self.ITER}: FRAME {frame_idx + 1}/{num_frames}: "
                f"Dx={dx:.3f}, Dy={dy:.3f}, Fluence={fluence:.3f}, Error={error:.3e}",
                flush=True
            )

        fitted_dx = cp.array([res['dx'] for res in results])
        fitted_dy = cp.array([res['dy'] for res in results])
        fitted_fluence = cp.array([res['fluence'] for res in results])
        min_errors = cp.array([res['error'] for res in results])

        return fitted_dx, fitted_dy, fitted_fluence, min_errors

