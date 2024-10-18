import cupy as cp
import h5py
import sys
from cupyx.scipy.ndimage import map_coordinates
from utils import get_vals

class ObjectOptimizer:
    def __init__(self, itern, N, shifts, fluence, ftobj, data_file, output_file):
        self.N = N
        self.cen = N // 2

        self.DATA_FILE = data_file
        self.OUTPUT_FILE = output_file

        self.ITER = itern
        self.shifts = cp.asarray(shifts)
        self.fluence_vals = cp.asarray(fluence)
        self.prev_ftobj = cp.asarray(ftobj)

        self.load_dataset()
        self.SWITCH_TO_ALLPIX = 50

    def load_dataset(self):
        # Load Dataset
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])
            self.num_samples = self.intens_vals.shape[0]

    def rotate_ft(self, ftobj, angle_deg):
        """ Rotate the Fourier transform object using scipy's map_coordinates """
        angle_rad = cp.deg2rad(angle_deg)
        qh, qk = cp.indices((self.N, self.N))
        qh = qh - self.cen
        qk = qk - self.cen
        qh_rot = cp.cos(angle_rad) * qh - cp.sin(angle_rad) * qk
        qk_rot = cp.sin(angle_rad) * qh + cp.cos(angle_rad) * qk
        coords = cp.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(cp.abs(ftobj).get(), coords.get(), order=3, mode='wrap')
        return cp.asarray(rotated_ft)

    # GRID SEARCH for each pixel
    def optimize_pixel(self, h, k, angle):
        funitc_val = get_vals(self.funitc, self.cen, h, k)
        intens_vals = self.intens_vals[:, h + self.cen, k + self.cen]
        phases = 2 * cp.pi * (h * self.shifts[:, 0] + k * self.shifts[:, 1])
        pramp = cp.exp(1j * phases)

        rotated_ftobj = self.rotate_ft(self.prev_ftobj, angle)  # Apply rotation

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
        for itr in range(2000):
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

        return best_ftobj, itr

    def optimize_all_pixels(self, angle):
        if self.ITER < self.SWITCH_TO_ALLPIX:
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
        itrs = []
        for idx in range(total_pixels):
            h = int(h_indices[idx])
            k = int(k_indices[idx])
            best_ftobj, itr = self.optimize_pixel(h, k, angle)
            optimized_params[h + self.cen, k + self.cen] = best_ftobj
            self._print_progress(idx + 1, total_pixels)
            itrs.append(itr)

        return optimized_params, cp.array(itrs)

    def _print_progress(self, count, total):
        progress = (count / total) * 100
        print(f"\rOptimized {count}/{total} pixels ({progress:.2f}%)", end='', flush=True)

