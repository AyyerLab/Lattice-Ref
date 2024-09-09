import cupy as cp
import h5py
import sys
from utils import get_vals  # Assuming this is already GPU-optimized

class ObjectOptimizer:
    def __init__(self, N, DATA_FILE, shifts, fluence_vals, OUTPUT_FILE, itern):
        self.N = N
        self.cen = self.N // 2
        self.ITERATION = itern
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE

        self.shifts = cp.asarray(shifts)
        self.fluence_vals = cp.asarray(fluence_vals)

        # Hyperparameters in uppercase
        self.INIT_FTOBJ_REAL = (-2000, 2000)
        self.INIT_FTOBJ_IMAG = (-2000, 2000)
        self.INIT_NUM_POINTS = 200

        self.MAX_REFINEMENTS = 100
        self.REFINEMENT_STEPS = 1000
        self.THRESHOLD = 1e-8
        self.INIT_GSIZE = 200

        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'][:])
            self.funitc = cp.asarray(f['funitc'][:])
            self.NUM_SAMPLES = self.intens_vals.shape[0]

    def optimize_pixel(self, qh, qk):
        funitc_pixvals = get_vals(self.funitc, self.cen, qh, qk)
        intens_pixvals = cp.array([get_vals(self.intens_vals[i], self.cen, qh, qk) for i in range(self.NUM_SAMPLES)])

        phases = 2.0 * cp.pi * (qh * self.shifts[:, 0] + qk * self.shifts[:, 1])
        pramp = cp.exp(1j * phases)

        def vectorized_objective(real_vals, imag_vals):
            real_vals = real_vals[:, cp.newaxis, cp.newaxis]
            imag_vals = imag_vals[:, cp.newaxis, cp.newaxis]
            ftobj_val = real_vals + 1j * imag_vals

            F = funitc_pixvals + self.fluence_vals * ftobj_val * pramp
            model_int = cp.abs(F)**2
            residuals = model_int - intens_pixvals
            error = cp.sum(residuals**2, axis=(1, 2))
            return error

        def coarse_search():
            real_vals = cp.linspace(self.INIT_FTOBJ_REAL[0], self.INIT_FTOBJ_REAL[1], self.INIT_NUM_POINTS)
            imag_vals = cp.linspace(self.INIT_FTOBJ_IMAG[0], self.INIT_FTOBJ_IMAG[1], self.INIT_NUM_POINTS)

            real_grid, imag_grid = cp.meshgrid(real_vals, imag_vals)
            real_flat = real_grid.ravel()
            imag_flat = imag_grid.ravel()

            errors = vectorized_objective(real_flat, imag_flat)
            min_index = cp.argmin(errors)
            best_real = real_flat[min_index]
            best_imag = imag_flat[min_index]
            min_error = errors[min_index]

            return best_real, best_imag, min_error

        def refinement_search(best_real, best_imag, min_error):
            gsize = self.INIT_GSIZE

            for _ in range(self.MAX_REFINEMENTS):
                real_fine_range = cp.linspace(best_real - gsize, best_real + gsize, 10)
                imag_fine_range = cp.linspace(best_imag - gsize, best_imag + gsize, 10)

                real_fine_grid, imag_fine_grid = cp.meshgrid(real_fine_range, imag_fine_range)
                real_fine_flat = real_fine_grid.ravel()
                imag_fine_flat = imag_fine_grid.ravel()

                errors = vectorized_objective(real_fine_flat, imag_fine_flat)
                min_fine_index = cp.argmin(errors)
                best_real_fine = real_fine_flat[min_fine_index]
                best_imag_fine = imag_fine_flat[min_fine_index]
                min_error_fine = errors[min_fine_index]

                if cp.abs(min_error - min_error_fine) < self.THRESHOLD:
                    break

                best_real, best_imag = best_real_fine, best_imag_fine
                min_error = min_error_fine
                gsize /= 2

            return best_real, best_imag

        best_real, best_imag, min_error = coarse_search()
        best_real, best_imag = refinement_search(best_real, best_imag, min_error)

        return best_real.get(), best_imag.get()

    def optimize_all_pixels(self):
        optimized_params = cp.zeros((self.N, self.N, 2))  # Assuming the last dimension is for real and imaginary parts
        total_pixels = self.N * self.N
        pixel_count = 0

        for h in range(-self.cen, self.cen + 1):
            for k in range(-self.cen, self.cen + 1):
                best_real, best_imag = self.optimize_pixel(h, k)  # Unpack the tuple here
                optimized_params[h + self.cen, k + self.cen, 0] = best_real  # Assign real part
                optimized_params[h + self.cen, k + self.cen, 1] = best_imag  # Assign imaginary part
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
            f.create_dataset('fitted_dx', data=cp.asnumpy(self.shifts[:, 0]))
            f.create_dataset('fitted_dy', data=cp.asnumpy(self.shifts[:, 1]))
            f.create_dataset('fitted_fluence', data=cp.asnumpy(self.fluence_vals))
            f.create_dataset('ftobj_fitted', data=cp.asnumpy(optimized_params))

