import cupy as cp
import h5py
import argparse
from configparser import ConfigParser
from utils_cupy import rotate_arr


class OrientOptimizer:
    def __init__(self, N, fluence, shifts, ftobj, data_file, fine_step_size=0.001):
        self.N = N
        self.cen = self.N // 2

        self.qh, self.qk = cp.indices((N, N))
        self.qh -= self.cen
        self.qk -= self.cen
        self.hk = cp.array([(h, k) for h, k in zip(self.qh.flatten(), self.qk.flatten()) if h**2 + k**2 < 45**2])

        self.data_file = data_file

        self.fluence = cp.asarray(fluence)
        self.NUM_SAMPLES = len(self.fluence)
        self.shifts = cp.asarray(shifts)
        self.ftobj = cp.asarray(ftobj)

        self.load_dataset()
        self.fine_step_size = fine_step_size

    def load_dataset(self):
        with h5py.File(self.data_file, 'r') as f:
            self.funitc = cp.asarray(f['funitc'][:])
            self.intens_vals = cp.asarray(f['intens'][:])

    def optimize_orientation(self):
        optimal_angs = cp.zeros(self.NUM_SAMPLES)
        steps = cp.zeros(self.NUM_SAMPLES)

        hk_indices = (self.hk[:, 0] + self.cen, self.hk[:, 1] + self.cen)

        angle_threshold = 1e-4

        for i in range(self.NUM_SAMPLES):
            dx = self.shifts[:, 0][i]
            dy = self.shifts[:, 1][i]
            fluence = self.fluence[i]
            intens_sample = self.intens_vals[i]

            funitc_vals = self.funitc[hk_indices]
            intens_vals_sample = intens_sample[hk_indices]

            def objective_ang(ang):
                rotated_ftobj = rotate_arr(self.N, self.ftobj, ang)

                qh = self.hk[:, 0]
                qk = self.hk[:, 1]
                phase = 2.0 * cp.pi * (qh * dx + qk * dy)
                pramp = cp.exp(1j * phase)

                rotated_vals = rotated_ftobj[hk_indices]
                model_int = cp.abs(funitc_vals + fluence * rotated_vals * pramp)**2
                error = cp.sum((model_int - intens_vals_sample) ** 2)
                return error

            # Coarse Search
            ncoarse = 360
            ang_coarse_range = cp.linspace(0, 2 * cp.pi, ncoarse)

            objective_values_coarse = cp.array([objective_ang(ang) for ang in ang_coarse_range])
            min_error_coarse = cp.min(objective_values_coarse)
            optimal_ang_coarse = ang_coarse_range[cp.argmin(objective_values_coarse)]

            coarse_steps = len(ang_coarse_range)
            total_steps = coarse_steps

            # Refinement
            current_ang = optimal_ang_coarse
            min_error_current = min_error_coarse

            fine_step_size = self.fine_step_size
            previous_ang = float('inf')

            while cp.abs(current_ang - previous_ang) > angle_threshold:

                lower_bound = cp.maximum(current_ang - 0.02, 0)
                upper_bound = cp.minimum(current_ang + 0.02, 2 * cp.pi)
                ang_fine_range = cp.arange(lower_bound, upper_bound, fine_step_size)

                fine_objective_values = cp.array([objective_ang(ang) for ang in ang_fine_range])
                min_error_fine = cp.min(fine_objective_values)
                optimal_ang_fine = ang_fine_range[cp.argmin(fine_objective_values)]

                previous_ang = current_ang
                current_ang = optimal_ang_fine
                min_error_current = min_error_fine

                total_steps += len(ang_fine_range)
                fine_step_size /= 2.0

            optimal_angs[i] = current_ang
            steps[i] = total_steps

            print(f"Frame {i}:  Best Angle: {current_ang}, Steps: {total_steps}, Error: {min_error_current}", flush=True)

        return optimal_angs, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize orientation using OrientOptimizer.")
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    N = config.getint("PARAMETERS", "n")
    data_file = config["FILES"]["data_file"]

    with h5py.File(data_file, "r") as f:
        ftobj = cp.asarray(f["ftobj"][:])
        shifts = cp.asarray(f["shifts"][:])
        fluence = cp.asarray(f["fluence"][:])

    fine_step_size = 0.001 

    optimizer = OrientOptimizer(N, fluence, shifts, ftobj, data_file, fine_step_size)
    optimal_angs, steps = optimizer.optimize_orientation()
    with h5py.File('/scratch/mallabhi/lattice_ref/output/optimize_orient_RIB_10Angs.h5', "w") as f:
        f['fitted_angles'] = optimal_angs.get()
    print("Orientation Optimization Complete.")

