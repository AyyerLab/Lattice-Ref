import numpy as np
import h5py
import sys
import configparser
from scipy import ndimage

from utils import get_vals

class ObjectOptimizer:
    def __init__(self, N, DATA_FILE, OUTPUT_FILE, INIT_FTOBJ_TYPE, PIXELS):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE
        self.INIT_FTOBJ_TYPE = INIT_FTOBJ_TYPE
        self.PIXELS = PIXELS

        self.load_dataset()
        self.get_ftobj()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = np.asarray(f['intens'])
            self.funitc = np.asarray(f['funitc'])
            self.shifts = np.asarray(f['shifts'])
            self.fluence_vals = np.asarray(f['fluence'])
            self.angles = np.asarray(f['angle'])  # Fix to match dataset key
            self.num_samples = self.intens_vals.shape[0]

    def do_fft(self, obj):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))

    def init_ftobj(self):
        size = (self.N, self.N)
        obj = np.zeros(size)
        cen = (self.N // 2, self.N // 2)
        y, x = np.ogrid[:self.N, :self.N]
        dcen = np.sqrt((x - cen[1])**2 + (y - cen[0])**2)
        obj.ravel()[np.argsort(dcen.ravel())[:self.PIXELS]] = 1
        self.ftobj = self.do_fft(obj)

    def get_ftobj(self):
        if self.INIT_FTOBJ_TYPE == 'RD':
            self.ftobj = np.random.rand(self.N, self.N) + 1j * np.random.rand(self.N, self.N)
        elif self.INIT_FTOBJ_TYPE == 'TS':
            with h5py.File(self.DATA_FILE, 'r') as file:
                self.ftobj = np.asarray(file['ftobj'][:])
        elif self.INIT_FTOBJ_TYPE == 'CR':
            self.init_ftobj()
        else:
            raise ValueError(f"Unknown INIT_FTOBJ_TYPE: {self.INIT_FTOBJ_TYPE}")

    def rotate_ft(self, ftobj, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        qh, qk = np.indices((self.N, self.N))
        qh -= self.cen
        qk -= self.cen

        qh_rot = np.cos(angle_rad) * qh - np.sin(angle_rad) * qk
        qk_rot = np.sin(angle_rad) * qh + np.cos(angle_rad) * qk

        coords = np.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = ndimage.map_coordinates(np.abs(ftobj), coords, order=3, mode='wrap')
        return rotated_ft

    def optimize_pixel(self, h, k):
        funitc_val = get_vals(self.funitc, self.cen, h, k)
        intens_vals = self.intens_vals[:, h + self.cen, k + self.cen]
        phases = 2 * np.pi * (h * self.shifts[:, 0] + k * self.shifts[:, 1])
        pramp = np.exp(1j * phases)

        def objective(theta):
            # Rotate the full ftobj, not just a single pixel
            rotated_ftobj = self.rotate_ft(self.ftobj, theta)
            rotated_val = rotated_ftobj[h + self.cen, k + self.cen]  # Extract the value after rotation
            F = funitc_val + self.fluence_vals[:, np.newaxis] * rotated_val * pramp[:, np.newaxis]
            model_int = np.abs(F) ** 2
            residuals = model_int - intens_vals[:, np.newaxis]
            return np.sum(residuals ** 2, axis=0)

        # Coarse grid search for orientation
        ncoarse = 20
        theta_coarse_range = np.linspace(0, 180, ncoarse)
        min_error = np.finfo('f8').max
        optimal_theta = 0

        for theta in theta_coarse_range:
            error = objective(theta)
            if error < min_error:
                min_error = error
                optimal_theta = theta

        # Fine optimization of theta
        theta_fine_range = np.arange(max(optimal_theta - 5, 0), min(optimal_theta + 5, 180), 0.1)
        min_error_fine = np.finfo('f8').max
        best_ftobj = None

        for theta in theta_fine_range:
            error = objective(theta)
            if error < min_error_fine:
                min_error_fine = error
                best_ftobj = self.ftobj[h + self.cen, k + self.cen]  # Store the best ftobj value

        return best_ftobj

    def optimize_all_pixels(self):
        h_indices = np.arange(-self.cen, self.cen + 1)
        k_indices = np.arange(-self.cen, self.cen + 1)
        h_indices, k_indices = np.meshgrid(h_indices, k_indices)
        h_indices = h_indices.ravel()
        k_indices = k_indices.ravel()

        total_pixels = h_indices.size
        optimized_params = self.ftobj.copy()

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
        output_file = self.OUTPUT_FILE  # Removed the iteration replacement logic

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('ftobj_fitted', data=np.asarray(optimized_params))

        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    N = int(config['DATA_GENERATION']['N'])
    DATA_FILE = config['DATA_GENERATION']['DATA_FILE']
    OUTPUT_FILE = config['OPTIMIZATION']['OUTPUT_FILE']
    INIT_FTOBJ_TYPE = config['OPTIMIZATION']['INIT_FTOBJ']
    PIXELS = int(config['OPTIMIZATION']['PIXELS'])

    optimizer = ObjectOptimizer(N, DATA_FILE, OUTPUT_FILE, INIT_FTOBJ_TYPE, PIXELS)
    optimizer.optimize_all_pixels()

