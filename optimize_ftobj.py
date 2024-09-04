import numpy as np
import h5py
import sys
from utils import get_vals

class ObjectOptimizer:
    def __init__(self, N, DATA_FILE, shifts, fluence_vals, OUTPUT_FILE, itern):
        self.N = N
        self.cen = self.N // 2
        self.ITERATION = itern
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE

        self.shifts = shifts
        self.fluence_vals = fluence_vals

        # Hyperparameters in uppercase
        self.INIT_FTOBJ_REAL = (-2000, 2000)
        self.INIT_FTOBJ_IMAG = (-2000, 2000)
        self.INIT_NUM_POINTS = 200

        self.MAX_REFINEMENTS = 100
        self.REFINEMENT_STEPS = 1000
        self.THRESHOLD = 1e-8
        self.INIT_GSIZE = 200  # Updated INIT_gsize to INIT_GSIZE

        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.funitc = f['funitc'][:]
            self.NUM_SAMPLES = self.intens_vals.shape[0]

    def optimize_pixel(self, qh, qk):
        funitc_pixvals = get_vals(self.funitc, self.cen, qh, qk)
        intens_pixvals = np.array([get_vals(self.intens_vals[i], self.cen, qh, qk) for i in range(self.NUM_SAMPLES)])

        phases = 2.0 * np.pi * (qh * self.shifts[:, 0] + qk * self.shifts[:, 1])
        pramp = np.exp(1j * phases)

        def vectorized_objective(real_vals, imag_vals):
            real_vals = real_vals[:, np.newaxis, np.newaxis]
            imag_vals = imag_vals[:, np.newaxis, np.newaxis]
            ftobj_val = real_vals + 1j * imag_vals

            F = funitc_pixvals + self.fluence_vals * ftobj_val * pramp
            model_int = np.abs(F)**2
            residuals = model_int - intens_pixvals
            error = np.sum(residuals**2, axis=(1, 2))
            return error

        def coarse_search():
            real_vals = np.linspace(self.INIT_FTOBJ_REAL[0], self.INIT_FTOBJ_REAL[1], self.INIT_NUM_POINTS)
            imag_vals = np.linspace(self.INIT_FTOBJ_IMAG[0], self.INIT_FTOBJ_IMAG[1], self.INIT_NUM_POINTS)

            real_grid, imag_grid = np.meshgrid(real_vals, imag_vals)
            real_flat = real_grid.ravel()
            imag_flat = imag_grid.ravel()

            errors = vectorized_objective(real_flat, imag_flat)
            min_index = np.argmin(errors)
            best_real = real_flat[min_index]
            best_imag = imag_flat[min_index]
            min_error = errors[min_index]

            return best_real, best_imag, min_error

        def refinement_search(best_real, best_imag, min_error):
            gsize = self.INIT_GSIZE

            for _ in range(self.MAX_REFINEMENTS):
                real_fine_range = np.linspace(best_real - gsize, best_real + gsize, 10)
                imag_fine_range = np.linspace(best_imag - gsize, best_imag + gsize, 10)

                real_fine_grid, imag_fine_grid = np.meshgrid(real_fine_range, imag_fine_range)
                real_fine_flat = real_fine_grid.ravel()
                imag_fine_flat = imag_fine_grid.ravel()

                errors = vectorized_objective(real_fine_flat, imag_fine_flat)
                min_fine_index = np.argmin(errors)
                best_real_fine = real_fine_flat[min_fine_index]
                best_imag_fine = imag_fine_flat[min_fine_index]
                min_error_fine = errors[min_fine_index]

                if np.abs(min_error - min_error_fine) < self.THRESHOLD:
                    break

                best_real, best_imag = best_real_fine, best_imag_fine
                min_error = min_error_fine
                gsize /= 2

            return best_real, best_imag

        best_real, best_imag, min_error = coarse_search()
        best_real, best_imag = refinement_search(best_real, best_imag, min_error)

        return best_real, best_imag

    def optimize_all_pixels(self):
        optimized_params = np.zeros((self.N, self.N, 2))
        total_pixels = (self.N * self.N)
        pixel_count = 0

        for h in range(-self.cen, self.cen + 1):
            for k in range(-self.cen, self.cen + 1):
                params = self.optimize_pixel(h, k)
                optimized_params[h + self.cen, k + self.cen] = params
                pixel_count += 1
                self._print_progress(pixel_count, total_pixels)

        self.save_results(optimized_params, self.ITERATION)

        return optimized_params

    def _print_progress(self, count, total):
        PROGRESS = (count / total) * 100
        sys.stdout.write(f'\rOPTIMIZED {count}/{total} PIXELS ({PROGRESS:.2f}%)')
        sys.stdout.flush()

    def save_results(self, optimized_params, iteration):
        output_file = self.OUTPUT_FILE.replace('.h5', f'{iteration}.h5')

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('fitted_dx', data=self.shifts[:, 0])
            f.create_dataset('fitted_dy', data=self.shifts[:, 1])
            f.create_dataset('fitted_fluence', data=self.fluence_vals)
            f.create_dataset('ftobj_fitted', data=optimized_params)

