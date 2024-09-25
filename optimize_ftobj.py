import cupy as cp
import h5py
import sys

from utils import get_vals

class ObjectOptimizer:
    def __init__(self, N, DATA_FILE, shifts, fluence_vals, OUTPUT_FILE, iteration, prev_ftobj):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE
        self.iteration = iteration
        self.prev_ftobj = cp.asarray(prev_ftobj)
        self.shifts = cp.asarray(shifts)
        self.fluence_vals = cp.asarray(fluence_vals)

        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])
            self.num_samples = self.intens_vals.shape[0]

    def optimize_pixel(self, h, k):
        funitc_val = get_vals(self.funitc, self.cen, h, k)
        intens_vals = self.intens_vals[:, h + self.cen, k + self.cen]
        phases = 2 * cp.pi * (h * self.shifts[:, 0] + k * self.shifts[:, 1])
        pramp = cp.exp(1j * phases)

        def objective(ftobj_vals):
            F = funitc_val + self.fluence_vals[:, cp.newaxis] * ftobj_vals * pramp[:, cp.newaxis]
            model_int = cp.abs(F) ** 2
            residuals = model_int - intens_vals[:, cp.newaxis]
            return cp.sum(residuals ** 2, axis=0)

        # Coarse grid search
        real_vals = cp.linspace(-2000, 2000, 200)
        imag_vals = cp.linspace(-2000, 2000, 200)
        real_grid, imag_grid = cp.meshgrid(real_vals, imag_vals)
        ftobj_vals = (real_grid + 1j * imag_grid).ravel()

        errors = objective(ftobj_vals)
        min_idx = cp.argmin(errors)
        best_ftobj = ftobj_vals[min_idx]
        min_error = errors[min_idx]

        # Refinement
        gsize = 200
        threshold = 1e-8
        max_refinements = 1000

        for _ in range(max_refinements):
            real_range = cp.linspace(best_ftobj.real - gsize, best_ftobj.real + gsize, 10)
            imag_range = cp.linspace(best_ftobj.imag - gsize, best_ftobj.imag + gsize, 10)
            real_grid, imag_grid = cp.meshgrid(real_range, imag_range)
            ftobj_vals = (real_grid + 1j * imag_grid).ravel()

            errors = objective(ftobj_vals)
            min_idx = cp.argmin(errors)
            new_best_ftobj = ftobj_vals[min_idx]
            new_min_error = errors[min_idx]

            if cp.abs(min_error - new_min_error) < threshold:
                break

            best_ftobj = new_best_ftobj
            min_error = new_min_error
            gsize /= 2

        return best_ftobj

    def optimize_all_pixels(self):
        # Determine the pixels to optimize
        if self.iteration <= 50:
            radius = 30
            h_vals = cp.arange(-self.cen, self.cen + 1)
            k_vals = cp.arange(-self.cen, self.cen + 1)
            h_grid, k_grid = cp.meshgrid(h_vals, k_vals)
            mask = cp.sqrt(h_grid**2 + k_grid**2) <= radius
            h_indices = h_grid[mask]
            k_indices = k_grid[mask]
        else:
            h_indices = cp.arange(-self.cen, self.cen + 1)
            k_indices = cp.arange(-self.cen, self.cen + 1)
            h_indices, k_indices = cp.meshgrid(h_indices, k_indices)
            h_indices = h_indices.ravel()
            k_indices = k_indices.ravel()

        total_pixels = h_indices.size
        optimized_params = self.prev_ftobj.copy()

        for idx in range(total_pixels):
            h = int(h_indices[idx])
            k = int(k_indices[idx])
            best_ftobj = self.optimize_pixel(h, k)
            optimized_params[h + self.cen, k + self.cen] = best_ftobj
            self._print_progress(idx + 1, total_pixels)

        self.save_results(optimized_params)
        return optimized_params

    def _print_progress(self, count, total):
        progress = (count / total) * 100
        print(f"\rOptimized {count}/{total} pixels ({progress:.2f}%)", end='', flush=True)

    def save_results(self, optimized_params):
        output_file = self.OUTPUT_FILE.replace('.h5', f'{self.iteration:03}.h5')

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('fitted_dx', data=cp.asnumpy(self.shifts[:, 0]))
            f.create_dataset('fitted_dy', data=cp.asnumpy(self.shifts[:, 1]))
            f.create_dataset('fitted_fluence', data=cp.asnumpy(self.fluence_vals))
            f.create_dataset('ftobj_fitted', data=cp.asnumpy(optimized_params))

        print(f"\nResults saved to {output_file}")

