import cupy as cp
import configparser
import h5py
from cupyx.scipy.ndimage import map_coordinates

class OrientOptimizer:
    def __init__(self, N, fluence, shifts, ftobj, data_file):
        self.N = N
        self.cen = self.N // 2

        self.qh, self.qk = cp.indices((N, N))
        self.qh -= self.cen
        self.qk -= self.cen
        self.hk = cp.array([(h, k) for h, k in zip(self.qh.flatten(), self.qk.flatten()) if h**2 + k**2 < 20**2])

        self.DATA_FILE = data_file

        self.FLUENCE = cp.asarray(fluence)
        self.NUM_SAMPLES = len(self.FLUENCE)
        self.SHIFTS = cp.asarray(shifts)
        self.ftobj = cp.asarray(ftobj)

        self.load_dataset()
        self.fine_step_size = 0.01

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.funitc = cp.asarray(f['funitc'][:])
            self.intens_vals = cp.asarray(f['intens'][:])

    def rotate_ft(self, ftobj, angle_deg):
        angle_rad = cp.deg2rad(angle_deg)
        qh_rot = cp.cos(angle_rad) * self.qh - cp.sin(angle_rad) * self.qk
        qk_rot = cp.sin(angle_rad) * self.qh + cp.cos(angle_rad) * self.qk
        coords = cp.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(ftobj, coords, order=3, mode='wrap')
        return rotated_ft

    def _getvals(self, array, h, k):
        qh = h + self.cen
        qk = k + self.cen
        return array[qh, qk]

    def optimize_orientation(self):
        optimal_thetas = cp.zeros(self.NUM_SAMPLES)
        steps = cp.zeros(self.NUM_SAMPLES)

        hk_indices = (self.hk[:, 0] + self.cen, self.hk[:, 1] + self.cen)

        for i in range(self.NUM_SAMPLES):
            dx = self.SHIFTS[:, 0][i]
            dy = self.SHIFTS[:, 1][i]
            fluence = self.FLUENCE[i]
            intens_sample = self.intens_vals[i]

            funitc_vals = self.funitc[hk_indices]
            intens_vals_sample = intens_sample[hk_indices]

            def objective_theta(theta):
                rotated_ftobj = self.rotate_ft(self.ftobj, theta)
                qh = self.hk[:, 0]
                qk = self.hk[:, 1]
                phase = 2.0 * cp.pi * (qh * dx + qk * dy)
                pramp = cp.exp(1j * phase)

                rotated_vals = rotated_ftobj[hk_indices]
                model_int = cp.abs(funitc_vals + fluence * rotated_vals * pramp)**2
                error = cp.sum((model_int - intens_vals_sample) ** 2)
                return error

            # Coarse optimization
            ncoarse = 20
            theta_coarse_range = cp.linspace(0, 180, ncoarse)
            min_error = cp.finfo('f8').max
            optimal_theta = 0
            coarse_steps = 0

            # Parallel evaluation of objective function for coarse optimization
            objective_values = cp.array([objective_theta(theta) for theta in theta_coarse_range])
            min_error = cp.min(objective_values)
            optimal_theta = theta_coarse_range[cp.argmin(objective_values)]
            coarse_steps = len(theta_coarse_range)

            # Fine optimization
            theta_fine_range = cp.arange(max(optimal_theta - 5, 0),
                                         min(optimal_theta + 5, 180),
                                         self.fine_step_size)
            fine_steps = len(theta_fine_range)

            # Parallel evaluation of objective function for fine optimization
            fine_objective_values = cp.array([objective_theta(theta) for theta in theta_fine_range])
            min_error_fine = cp.min(fine_objective_values)
            optimal_theta_fine = theta_fine_range[cp.argmin(fine_objective_values)]

            optimal_thetas[i] = optimal_theta_fine
            total_steps = coarse_steps + fine_steps
            steps[i] = total_steps
            print(f"Frame {i}:  Theta: {optimal_theta_fine}, Steps: {total_steps}", flush=True)

        return optimal_thetas, steps

