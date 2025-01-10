import numpy as np
from scipy.ndimage import map_coordinates
from scipy import ndimage
import h5py

def do_fft(obj):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

def rotate_ft(N, ftobj, angle):
    cen = N // 2
    qh, qk = np.indices((N, N))
    qk -= cen
    qh -= cen

    qh_rot = np.cos(angle) * qh - np.sin(angle) * qk
    qk_rot = np.sin(angle) * qh + np.cos(angle) * qk
    coords = np.array([qh_rot + cen, qk_rot + cen])
    rotated_ft = map_coordinates(ftobj, coords, order=1, mode='nearest')
    return rotated_ft

def phase_ramp(N, shiftx, shifty):
    cen = N // 2
    qh, qk = np.indices((N, N))
    qh -= cen
    qk -= cen
    return np.exp(1j * 2.0 * np.pi * (qh * shiftx + qk * shifty))

def _getvals(array, N, Qh, Qk):
    cen = N // 2
    h_index = Qh + cen
    k_index = Qk + cen
    if 0 <= h_index < N and 0 <= k_index < N:
        return array[h_index, k_index]
    return 0


class ObjectOptimizer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.load_data()

        self.coarse_size = 100
        self.grid_size = 40
        self.initial_range = 20.0
        self.range_decay = 0.5
        self.convergence_threshold = 1e-5
        self.max_refinement_steps = 100

        self.cen = self.N // 2
        self.qh, self.qk = np.indices((self.N, self.N))
        self.qk -= self.cen
        self.qh -= self.cen

    def load_data(self):
        with h5py.File(self.data_file, "r") as f:
            self.tobj = f["tobj"][:]         # shape: (N, N)
            self.ftobj = f["ftobj"][:]       # shape: (N, N)
            self.unitc = f["unitc"][:]       # shape: (N, N)
            self.funitc = f["funitc"][:]     # shape: (N, N)
            self.intens = f["intens"][:]     # shape: (num_frames, N, N)
            self.angles = f["angles"][:]     # shape: (num_frames,)
            self.fluence = f["fluence"][:]   # shape: (num_frames,)
            self.shifts = f["shifts"][:]     # shape: (num_frames, 2)

        self.shiftx = self.shifts[:, 0]
        self.shifty = self.shifts[:, 1]

        self.N = self.tobj.shape[0]
        self.num_frames = self.intens.shape[0]

    def compute_error_grid(self, real_range, imag_range,
                           funitc_vals, fluence_vals, pramp_vals, intens_vals):
        real_grid, imag_grid = np.meshgrid(real_range, imag_range, indexing='ij')
        ftobj_guess_grid = real_grid + 1j * imag_grid

        res = (funitc_vals[:, None, None] +
               fluence_vals[:, None, None] *
               ftobj_guess_grid[None, :, :] *
               pramp_vals[:, None, None])

        Icalc = np.abs(res)**2
        err = (Icalc - intens_vals[:, None, None])**2
        return err.sum(axis=0)

    def solve(self):
        results = np.zeros((self.N, self.N), dtype=complex)
        total_pixels = (2 * self.cen + 1)**2
        pixel_counter = 0

        for Qh_ in range(-self.cen, self.cen + 1):
            for Qk_ in range(-self.cen, self.cen + 1):
                pixel_counter += 1
                print(f"\rProcessing pixel {pixel_counter}/{total_pixels} "
                      f"(Qh={Qh_}, Qk={Qk_})", end="")

                funitc_vals = []
                intens_vals = []
                pramp_vals = []

                for i in range(self.num_frames):
                    angle = -self.angles[i]
                    qh_r = np.cos(angle) * Qh_ - np.sin(angle) * Qk_
                    qk_r = np.sin(angle) * Qh_ + np.cos(angle) * Qk_

                    qh_r_rounded = int(round(qh_r))
                    qk_r_rounded = int(round(qk_r))

                    funitc_vals.append(_getvals(self.funitc, self.N,
                                                qh_r_rounded, qk_r_rounded))
                    intens_vals.append(_getvals(self.intens[i], self.N,
                                                qh_r_rounded, qk_r_rounded))
                    pramp_vals.append(
                        np.exp(1j * 2*np.pi * (
                            qh_r_rounded * self.shiftx[i] +
                            qk_r_rounded * self.shifty[i]
                        ))
                    )

                funitc_vals = np.array(funitc_vals)
                intens_vals = np.array(intens_vals)
                pramp_vals = np.array(pramp_vals)

                coarse_real_range = np.linspace(-3000, 3000, self.coarse_size)
                coarse_imag_range = np.linspace(-3000, 3000, self.coarse_size)
                err_grid_coarse = self.compute_error_grid(coarse_real_range,
                                                          coarse_imag_range,
                                                          funitc_vals,
                                                          self.fluence,
                                                          pramp_vals,
                                                          intens_vals)

                min_index_coarse = np.unravel_index(np.argmin(err_grid_coarse),
                                                    err_grid_coarse.shape)
                coarse_best_real = coarse_real_range[min_index_coarse[0]]
                coarse_best_imag = coarse_imag_range[min_index_coarse[1]]

                current_best_real = coarse_best_real
                current_best_imag = coarse_best_imag
                current_range = self.initial_range
                prev_fitted_value = coarse_best_real + 1j * coarse_best_imag

                for step in range(self.max_refinement_steps):
                    fine_real_range = np.linspace(
                        current_best_real - current_range,
                        current_best_real + current_range,
                        self.grid_size
                    )
                    fine_imag_range = np.linspace(
                        current_best_imag - current_range,
                        current_best_imag + current_range,
                        self.grid_size
                    )

                    err_grid_fine = self.compute_error_grid(
                        fine_real_range,
                        fine_imag_range,
                        funitc_vals,
                        self.fluence,
                        pramp_vals,
                        intens_vals
                    )

                    min_index_fine = np.unravel_index(np.argmin(err_grid_fine),
                                                      err_grid_fine.shape)
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

                results[Qh_ + self.cen, Qk_ + self.cen] = fitted_value

        print("\n")
        return results


if __name__ == "__main__":
    data_file = "/scratch/mallabhi/lattice_ref/data/K/dataset_ori.h5"
    optimizer = ObjectOptimizer(data_file)
    result_grid = optimizer.solve()

    out_fname = "reconstructed_ftobj.h5"
    with h5py.File(out_fname, "w") as h5f:
        h5f.create_dataset("ftobj", data=result_grid)

    print(f"Reconstructed FTOBJ saved to {out_fname}")

