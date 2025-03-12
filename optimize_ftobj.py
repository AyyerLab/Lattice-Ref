import h5py
import argparse
import cupy as cp
from configparser import ConfigParser

class ObjectOptimizer:
    def __init__(self, itern, N, shifts, fluence, angles, data_file, output_file):
        self.N = N
        self.ITER = itern
        self.shiftx = shifts[:, 0]
        self.shifty = shifts[:, 1]
        self.fluence = fluence
        self.angles = angles

        self.data_file = data_file
        self.output_file = output_file
        self.load_data()

        self.cen = self.N // 2
        self.qh, self.qk = cp.meshgrid(cp.arange(self.N), cp.arange(self.N), indexing="ij")
        self.qk -= self.cen
        self.qh -= self.cen

        self.coarse_real_range = cp.linspace(-1200, 1200, 200)*10**(-4)
        self.coarse_imag_range = cp.linspace(-1200, 1200, 200)*10**(-4)

    def load_data(self):
        with h5py.File(self.data_file, "r") as f:
            self.funitc = cp.array(f["funitc"][:])
            self.intens = cp.array(f["intens"][:])

        self.NUM_FRAMES = self.intens.shape[0]

    def compute_error_grid(self, real_range, imag_range, funitc_vals, fluence_vals, pramp_vals, intens_vals):
        real_grid, imag_grid = cp.meshgrid(real_range, imag_range, indexing="ij")
        ftobj_guess_grid = real_grid + 1j * imag_grid

        Icalc = cp.abs(
            funitc_vals[:, None, None]
            + fluence_vals[:, None, None] * ftobj_guess_grid[None, :, :] * pramp_vals[:, None, None]
        ) ** 2

        err = (Icalc - intens_vals[:, None, None]) ** 2
        return err.sum(axis=0)

    def solve(self):
        results = cp.zeros((self.N, self.N), dtype=complex)
        total_pixels = (2 * self.cen + 1) ** 2
        pixel_counter = 0

        angles = -self.angles

        for Qh_ in range(-self.cen, self.cen + 1):
            for Qk_ in range(-self.cen, self.cen + 1):
                pixel_counter += 1
                print(f"\rProcessing pixel {pixel_counter}/{total_pixels} ", flush=True, end="")

                qh_r = cp.cos(angles) * Qh_ - cp.sin(angles) * Qk_
                qk_r = cp.sin(angles) * Qh_ + cp.cos(angles) * Qk_

                qh_r_rounded = cp.around(qh_r).astype(cp.int32)
                qk_r_rounded = cp.around(qk_r).astype(cp.int32)

                h_index = qh_r_rounded + self.cen
                k_index = qk_r_rounded + self.cen

                cp.clip(h_index, 0, self.N - 1, out=h_index)
                cp.clip(k_index, 0, self.N - 1, out=k_index)

                funitc_vals = self.funitc[h_index, k_index]
                intens_vals = self.intens[cp.arange(self.NUM_FRAMES), h_index, k_index]
                pramp_vals = cp.exp(1j * 2 * cp.pi * (qh_r_rounded * self.shiftx + qk_r_rounded * self.shifty))


                err_grid_coarse = self.compute_error_grid(
                    self.coarse_real_range,
                    self.coarse_imag_range,
                    funitc_vals,
                    self.fluence,
                    pramp_vals,
                    intens_vals,
                )

                min_index_coarse = cp.unravel_index(cp.argmin(err_grid_coarse), err_grid_coarse.shape)
                coarse_best_real = self.coarse_real_range[min_index_coarse[0]]
                coarse_best_imag = self.coarse_imag_range[min_index_coarse[1]]

                current_best_real = coarse_best_real
                current_best_imag = coarse_best_imag
                initial_range = 100*10**(-4)
                current_range = initial_range
                prev_fitted_value = coarse_best_real + 1j * coarse_best_imag

                grid_size = 40
                range_decay = 0.5
                convergence_threshold = 1e-5
                max_refinement_steps = 500

                for step in range(max_refinement_steps):
                    fine_real_range = cp.linspace(
                        current_best_real - current_range, current_best_real + current_range, grid_size
                    )
                    fine_imag_range = cp.linspace(
                        current_best_imag - current_range, current_best_imag + current_range, grid_size
                    )

                    err_grid_fine = self.compute_error_grid(
                        fine_real_range,
                        fine_imag_range,
                        funitc_vals,
                        self.fluence,
                        pramp_vals,
                        intens_vals,
                    )

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

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Fourier transform of an object.")
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    itern = config.getint("OPTIMIZATION", "num_iteration")
    N = config.getint("PARAMETERS", "N")
    data_file = config["FILES"]["data_file"]
    output_file = config["FILES"]["output_file"]
    with h5py.File(data_file, "r") as f:
        shifts = cp.asarray(f["shifts"][:])
        fluence = cp.asarray(f["fluence"][:])
        angles = cp.asarray(f["angles"][:])

    optimizer = ObjectOptimizer(itern, N, shifts, fluence, angles, data_file, output_file)
    results = optimizer.solve()

    with h5py.File('/scratch/mallabhi/lattice_ref/output/optimize_ftobj.h5', "w") as f:
        f['fitted_ftobj'] = results.get()
    print("\nOptimization Complete. Results saved.")

