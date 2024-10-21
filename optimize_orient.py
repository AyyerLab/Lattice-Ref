import cupy as cp
from cupyx.scipy.ndimage import map_coordinates
import configparser
import h5py

class OrientOptimizer:
    def __init__(self, N, fluence, shifts, ftobj, data_file):
        self.N = N
        self.cen = N // 2
        self.qh, self.qk = cp.indices((N, N))
        self.qh -= self.cen
        self.qk -= self.cen

        self.hk = cp.array([(0, 1), (1, 1), (1, 0), (1, -1)])

        self.DATA_FILE = data_file
        self.fluence = cp.asarray(fluence) 
        self.shifts = cp.asarray(shifts)
        self.ftobj = cp.asarray(ftobj)
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.funitc = cp.asarray(f['funitc'][:])
            self.intens_vals = cp.asarray(f['intens'][:])

        self.num_samples = len(self.fluence)
        self.fine_step_size = 0.01

    def rotate_ft(self, ftobj, angle_deg):
        angle_rad = cp.deg2rad(angle_deg)
        qh_rot = cp.cos(angle_rad) * self.qh - cp.sin(angle_rad) * self.qk
        qk_rot = cp.sin(angle_rad) * self.qh + cp.cos(angle_rad) * self.qk
        coords = cp.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(cp.abs(ftobj), coords, order=3, mode='wrap')
        return rotated_ft

    def _getvals(self, array, h, k):
        qh = h + self.cen
        qk = k + self.cen
        return array[qh, qk]


    def optimize_orientation(self):
        optimal_thetas = cp.zeros(self.num_samples)
        steps = cp.zeros(self.num_samples)

        for i in range(self.num_samples):
            dx = self.shifts[:, 0][i]
            dy = self.shifts[:, 1][i]
            fluence = self.fluence[i]
            intens_sample = self.intens_vals[i]

            funitc_vals = cp.array([self._getvals(self.funitc, h, k) for h, k in self.hk])
            intens_vals_sample = cp.array([self._getvals(intens_sample, h, k) for h, k in self.hk])

            def objective_theta(theta):
                rotated_ftobj = self.rotate_ft(self.ftobj, theta)
                qh = self.hk[:, 0]
                qk = self.hk[:, 1]
                phase = 2.0 * cp.pi * (qh * dx + qk * dy)
                pramp = cp.exp(1j * phase)

                rotated_vals = cp.array([self._getvals(rotated_ftobj, h, k) for h, k in self.hk])
                model_int = cp.abs(funitc_vals + fluence * rotated_vals * pramp)**2
                error = cp.sum((model_int - intens_vals_sample) ** 2)
                return error

            # Coarse optimization
            ncoarse = 20
            theta_coarse_range = cp.linspace(0, 180, ncoarse)
            min_error = cp.finfo('f8').max
            optimal_theta = 0
            coarse_steps = 0

            for theta in theta_coarse_range:
                objective_value = objective_theta(theta)
                coarse_steps += 1
                if objective_value < min_error:
                    min_error = objective_value
                    optimal_theta = theta

            # Fine optimization
            theta_fine_range = cp.arange(max(optimal_theta - 5, 0), min(optimal_theta + 5, 180), self.fine_step_size)
            min_error_fine = cp.finfo('f8').max
            optimal_theta_fine = optimal_theta
            fine_steps = 0

            for theta in theta_fine_range:
                objective_value = objective_theta(theta)
                fine_steps += 1 
                if objective_value < min_error_fine:
                    min_error_fine = objective_value
                    optimal_theta_fine = theta

            optimal_thetas[i] = optimal_theta_fine
            total_steps = coarse_steps + fine_steps
            steps[i] = total_steps
            print(f"Sample {i}: Optimal theta: {optimal_theta_fine}, Refinement steps: {total_steps}", flush=True)

        return optimal_thetas, steps

