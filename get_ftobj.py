import numpy as np
from scipy.ndimage import map_coordinates
from scipy import ndimage
import h5py
import mrcfile

# Explicit numpy implementations of FFT functions
def do_fft(obj):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

class ObjectOptimizer:
    def __init__(self, N, num_frames):
        self.N = N
        self.num_frames = num_frames
        self.coarse_size = 100
        self.grid_size = 40
        self.initial_range = 20.0
        self.range_decay = 0.5
        self.convergence_threshold = 1e-5
        self.max_refinement_steps = 100
        self.cen = N // 2
        self.qh, self.qk = np.indices((N, N))
        self.qk -= self.cen
        self.qh -= self.cen

    def target_obj(self):
        with mrcfile.open('/scratch/mallabhi/lattice_ref/data/PS1_map.mrc', permissive=True) as mrc:
            ps1 = mrc.data
        ps1_proj = ps1.sum(axis=0)
        pad_h = self.N - ps1_proj.shape[0]
        pad_w = self.N - ps1_proj.shape[1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_ps1_proj = np.pad(ps1_proj, ((pad_top, pad_bottom),
                                            (pad_left, pad_right)),
                                 mode='constant', constant_values=0)
        mask = np.asarray(padded_ps1_proj)
        return mask, do_fft(mask)

    def unit_cell(self):
        ind = np.arange(self.N) - self.N // 2
        rad = np.sqrt(ind[:, None]**2 + ind[None, :]**2)
        np.random.seed(42)
        random_ = np.random.rand(self.N, self.N) > 0.7
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap') * (rad < 30)
        return unitc, 100 * do_fft(unitc)

    def rotate_ft(self, ftobj, angle):
        qh_rot = np.cos(angle) * self.qh - np.sin(angle) * self.qk
        qk_rot = np.sin(angle) * self.qh + np.cos(angle) * self.qk
        coords = np.array([qh_rot + self.cen, qk_rot + self.cen])
        return map_coordinates(ftobj, coords, order=1, mode='nearest')

    def phase_ramp(self, shiftx, shifty):
        return np.exp(1j * 2.0 * np.pi * (self.qh * shiftx + self.qk * shifty))

    def _getvals(self, array, h, k):
        if 0 <= h + self.cen < self.N and 0 <= k + self.cen < self.N:
            return array[h + self.cen, k + self.cen]
        else:
            return 0

    def compute_error_grid(self, real_range, imag_range, funitc_vals, fluence_vals, pramp_vals, intens_vals):
        real_grid, imag_grid = np.meshgrid(real_range, imag_range, indexing='ij')
        ftobj_guess_grid = real_grid + 1j * imag_grid

        res = funitc_vals[:, None, None] + fluence_vals[:, None, None] * ftobj_guess_grid[None, :, :] * pramp_vals[:, None, None]
        Icalc = np.abs(res)**2
        err = (Icalc - intens_vals[:, None, None])**2
        return err.sum(axis=0)

    def solve(self):
        tobj, ftobj = self.target_obj()
        unitc, funitc = self.unit_cell()

        np.random.seed(42)
        dx_vals = np.random.uniform(0.01, 1, size=self.num_frames)
        dy_vals = np.random.uniform(0.01, 1, size=self.num_frames)
        fluence_vals = np.random.uniform(0.01, 1, size=self.num_frames)
        angle_vals = np.random.uniform(0, 1, size=self.num_frames) * 2 * np.pi

        intensities = np.zeros((self.num_frames, self.N, self.N))
        for i in range(self.num_frames):
            pramp = self.phase_ramp(dx_vals[i], dy_vals[i])
            intensities[i] = np.abs(funitc + fluence_vals[i] * self.rotate_ft(ftobj, angle_vals[i]) * pramp)**2

        Iobs = intensities
        results = np.zeros((self.N, self.N), dtype=complex)
        total_pixels = (2 * self.cen + 1)**2
        pixel_counter = 0

        for Qh in range(-self.cen, self.cen + 1):
            for Qk in range(-self.cen, self.cen + 1):
                pixel_counter += 1
                print(f"\rProcessing pixel {pixel_counter}/{total_pixels} (Qh={Qh}, Qk={Qk})", end="")

                ftobj_true_val = self._getvals(ftobj, Qh, Qk)
                funitc_vals = []
                intens_vals = []
                pramp_vals = []

                for i in range(self.num_frames):
                    angle = -angle_vals[i]
                    qh_r = np.cos(angle) * Qh - np.sin(angle) * Qk
                    qk_r = np.sin(angle) * Qh + np.cos(angle) * Qk
                    funitc_vals.append(self._getvals(funitc, round(qh_r), round(qk_r)))
                    intens_vals.append(self._getvals(Iobs[i], round(qh_r), round(qk_r)))
                    pramp_vals.append(np.exp(1j * 2 * np.pi * (round(qh_r) * dx_vals[i] + round(qk_r) * dy_vals[i])))

                funitc_vals = np.array(funitc_vals)
                intens_vals = np.array(intens_vals)
                pramp_vals = np.array(pramp_vals)

                coarse_real_range = np.linspace(-3000, 3000, self.coarse_size)
                coarse_imag_range = np.linspace(-3000, 3000, self.coarse_size)

                err_grid_coarse = self.compute_error_grid(coarse_real_range, coarse_imag_range, funitc_vals, fluence_vals, pramp_vals, intens_vals)
                min_index_coarse = np.unravel_index(np.argmin(err_grid_coarse), err_grid_coarse.shape)
                coarse_best_real = coarse_real_range[min_index_coarse[0]]
                coarse_best_imag = coarse_imag_range[min_index_coarse[1]]

                current_best_real = coarse_best_real
                current_best_imag = coarse_best_imag
                current_range = self.initial_range
                prev_fitted_value = coarse_best_real + 1j * coarse_best_imag

                for step in range(self.max_refinement_steps):
                    fine_real_range = np.linspace(current_best_real - current_range, current_best_real + current_range, self.grid_size)
                    fine_imag_range = np.linspace(current_best_imag - current_range, current_best_imag + current_range, self.grid_size)

                    err_grid_fine = self.compute_error_grid(fine_real_range, fine_imag_range, funitc_vals, fluence_vals, pramp_vals, intens_vals)
                    min_index_fine = np.unravel_index(np.argmin(err_grid_fine), err_grid_fine.shape)
                    fine_best_real = fine_real_range[min_index_fine[0]]
                    fine_best_imag = fine_imag_range[min_index_fine[1]]

                    current_best_real = fine_best_real
                    current_best_imag = fine_best_imag
                    fitted_value = current_best_real + 1j * current_best_imag
                    current_range *= self.range_decay

                    improvement = np.abs(fitted_value - prev_fitted_value)
                    if improvement < self.convergence_threshold:
                        break
                    prev_fitted_value = fitted_value

                results[Qh + self.cen, Qk + self.cen] = fitted_value

        print("\n")
        return results

if __name__ == "__main__":
    N = 101
    num_frames = 500
    optimizer = ObjectOptimizer(N, num_frames)
    result_grid = optimizer.solve()

    # Save the reconstructed FTOBJ to an H5 file
    with h5py.File("reconstructed_ftobj.h5", "w") as h5f:
        h5f.create_dataset("ftobj", data=result_grid)

    print("Reconstructed FTOBJ saved to reconstructed_ftobj.h5")

