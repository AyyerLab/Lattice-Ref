import cupy as cp
from cupyx.scipy.ndimage import map_coordinates
from cupyx.scipy import ndimage
import h5py

def do_fft(obj):
    return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(ftobj)))

def rotate_ft(N, ftobj, angle):
    cen = N // 2
    qh, qk = cp.meshgrid(cp.arange(N), cp.arange(N), indexing='ij')
    qk -= cen
    qh -= cen

    qh_rot = cp.cos(angle) * qh - cp.sin(angle) * qk
    qk_rot = cp.sin(angle) * qh + cp.cos(angle) * qk
    coords = cp.stack([qh_rot + cen, qk_rot + cen])

    rotated_ft = map_coordinates(ftobj, coords, order=1, mode='nearest')
    return rotated_ft

def phase_ramp(N, shiftx, shifty):
    cen = N // 2
    qh, qk = cp.meshgrid(cp.arange(N), cp.arange(N), indexing='ij')
    qh -= cen
    qk -= cen
    return cp.exp(1j * 2.0 * cp.pi * (qh * shiftx + qk * shifty))

def _getvals(array, N, Qh, Qk):
    cen = N // 2
    h_index = Qh + cen
    k_index = Qk + cen
    if 0 <= h_index < N and 0 <= k_index < N:
        return array[h_index, k_index]
    return 0

class ObjectOptimizer:
    def __init__(self, data_file, output_file):
        self.data_file = data_file
        self.output_file = output_file
        self.load_data()

        self.cen = self.N // 2
        self.qh, self.qk = cp.meshgrid(cp.arange(self.N), cp.arange(self.N), indexing='ij')
        self.qk -= self.cen
        self.qh -= self.cen

    def load_data(self):
        with h5py.File(self.data_file, "r") as f:
            self.ftobj = cp.array(f["ftobj"][:])
            self.funitc = cp.array(f["funitc"][:])
            self.intens = cp.array(f["intens"][:])
            self.angles = cp.array(f["angles"][:])
            self.fluence = cp.array(f["fluence"][:])
            self.shifts = cp.array(f["shifts"][:])

        self.shiftx = self.shifts[:, 0]
        self.shifty = self.shifts[:, 1]

        self.N = self.ftobj.shape[0]
        self.NUM_FRAMES = self.intens.shape[0]

    def compute_error_grid(self, real_range, imag_range,
                           funitc_vals, fluence_vals, pramp_vals, intens_vals):

        real_grid, imag_grid = cp.meshgrid(real_range, imag_range, indexing='ij')
        ftobj_guess_grid = real_grid + 1j * imag_grid

        Icalc = cp.abs(funitc_vals[:, None, None] + fluence_vals[:, None, None] *
                                                    ftobj_guess_grid[None, :, :] *
                                                    pramp_vals[:, None, None])**2

        err = (Icalc - intens_vals[:, None, None])**2
        return err.sum(axis=0)

    def solve(self):
        results = cp.zeros((self.N, self.N), dtype=complex)
        total_pixels = (2 * self.cen + 1)**2
        pixel_counter = 0

        for Qh_ in range(-self.cen, self.cen + 1):
            for Qk_ in range(-self.cen, self.cen + 1):
                pixel_counter += 1
                print(f"\rProcessing pixel {pixel_counter}/{total_pixels} ", end="")

                funitc_vals = cp.zeros(self.NUM_FRAMES, dtype=complex)
                intens_vals = cp.zeros(self.NUM_FRAMES, dtype=float)
                pramp_vals = cp.zeros(self.NUM_FRAMES, dtype=complex)

                for i in range(self.NUM_FRAMES):
                    angle = -self.angles[i]
                    qh_r = cp.cos(angle) * Qh_ - cp.sin(angle) * Qk_
                    qk_r = cp.sin(angle) * Qh_ + cp.cos(angle) * Qk_


                    qh_r_rounded = int(cp.around(qh_r).get())
                    qk_r_rounded = int(cp.around(qk_r).get())

                    funitc_vals[i] = _getvals(self.funitc, self.N, qh_r_rounded, qk_r_rounded)
                    intens_vals[i] = _getvals(self.intens[i], self.N, qh_r_rounded, qk_r_rounded)
                    pramp_vals[i] = cp.exp(1j * 2 * cp.pi * (qh_r_rounded * self.shiftx[i] +
                                                             qk_r_rounded * self.shifty[i]))

                coarse_size = 100
                initial_range = 20
                coarse_real_range = cp.linspace(-3000, 3000, coarse_size)
                coarse_imag_range = cp.linspace(-3000, 3000, coarse_size)
                err_grid_coarse = self.compute_error_grid(coarse_real_range, coarse_imag_range,
                                                          funitc_vals, self.fluence, pramp_vals, intens_vals)

                min_index_coarse = cp.unravel_index(cp.argmin(err_grid_coarse), err_grid_coarse.shape)
                coarse_best_real = coarse_real_range[min_index_coarse[0]]
                coarse_best_imag = coarse_imag_range[min_index_coarse[1]]

                current_best_real = coarse_best_real
                current_best_imag = coarse_best_imag
                current_range = initial_range
                prev_fitted_value = coarse_best_real + 1j * coarse_best_imag

                grid_size = 40
                range_decay = 0.5
                convergence_threshold = 1e-5
                max_refinement_steps = 100
                for step in range(max_refinement_steps):
                    fine_real_range = cp.linspace(current_best_real - current_range,
                                                  current_best_real + current_range, grid_size)
                    fine_imag_range = cp.linspace(current_best_imag - current_range,
                                                  current_best_imag + current_range, grid_size)

                    err_grid_fine = self.compute_error_grid(fine_real_range, fine_imag_range,
                                                            funitc_vals, self.fluence, pramp_vals, intens_vals)

                    min_index_fine = cp.unravel_index(cp.argmin(err_grid_fine), err_grid_fine.shape)
                    fine_best_real = fine_real_range[min_index_fine[0]]
                    fine_best_imag = fine_imag_range[min_index_fine[1]]

                    current_best_real = fine_best_real
                    current_best_imag = fine_best_imag
                    fitted_value = current_best_real + 1j * current_best_imag
                    current_range *= range_decay

                    improvement = cp.abs(fitted_value - prev_fitted_value)
                    if improvement < convergence_threshold:
                        break
                    prev_fitted_value = fitted_value

                results[Qh_ + self.cen, Qk_ + self.cen] = fitted_value

        print("\nSaving results to output file.")
        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("ftobj", data=cp.asnumpy(results))

        print("Reconstruction saved.")
        return results

if __name__ == "__main__":
    data_file = "/scratch/mallabhi/lattice_ref/data/K/dataset_ori.h5"
    output_file = "/scratch/mallabhi/lattice_ref/output/output_ori.h5"
    optimizer = ObjectOptimizer(data_file, output_file)
    optimizer.solve()

