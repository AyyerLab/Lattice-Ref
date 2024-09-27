import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
import configparser
import h5py

class OrientationOptimizer:
    def __init__(self, N, DATA_FILE):
        self.N = N
        self.cen = N // 2
        self.qh, self.qk = np.indices((N, N))
        self.qh -= self.cen
        self.qk -= self.cen

        self.hk = np.array([(0, 1), (1, 1), (1, 0), (1, -1)])  # Using array for vectorization

        self.DATA_FILE = DATA_FILE
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.fluence = f['fluence'][:]
            self.shifts = f['shifts'][:]
            self.ftobj = f['ftobj'][:]
            self.funitc = f['funitc'][:]
            self.intens_vals = f['intens'][:]

        self.num_samples = len(self.fluence)
        self.fine_step_size = 0.01

    def rotate_ft(self, ftobj, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        qh_rot = np.cos(angle_rad) * self.qh - np.sin(angle_rad) * self.qk
        qk_rot = np.sin(angle_rad) * self.qh + np.cos(angle_rad) * self.qk
        coords = np.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(np.abs(ftobj), coords, order=3, mode='wrap')
        return rotated_ft

    def _getvals(self, array, h, k):
        qh = h + self.cen
        qk = k + self.cen
        return array[qh, qk]

    def optimize_orientation(self):
        optimal_thetas = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            dx = self.shifts[:, 0][i]
            dy = self.shifts[:, 1][i]
            fluence = self.fluence[i]
            intens_sample = self.intens_vals[i]

            # Precompute constant values for this sample
            funitc_vals = np.array([self._getvals(self.funitc, h, k) for h, k in self.hk])
            intens_vals_sample = np.array([self._getvals(intens_sample, h, k) for h, k in self.hk])

            def objective_theta(theta):
                # Rotation and phase calculations
                rotated_ftobj = self.rotate_ft(self.ftobj, theta)
                qh = self.hk[:, 0]
                qk = self.hk[:, 1]
                phase = 2.0 * np.pi * (qh * dx + qk * dy)
                pramp = np.exp(1j * phase)

                # Compute model intensities
                rotated_vals = np.array([self._getvals(rotated_ftobj, h, k) for h, k in self.hk])
                model_int = np.abs(funitc_vals + fluence * rotated_vals * pramp)**2

                # Error calculation
                error = np.sum((model_int - intens_vals_sample) ** 2)
                return error

            # Coarse optimization
            ncoarse = 20
            theta_coarse_range = np.linspace(0, 180, ncoarse)
            min_error = np.finfo('f8').max
            optimal_theta = 0
            for theta in theta_coarse_range:
                objective_value = objective_theta(theta)
                if objective_value < min_error:
                    min_error = objective_value
                    optimal_theta = theta

            # Fine optimization
            theta_fine_range = np.arange(max(optimal_theta - 5, 0), min(optimal_theta + 5, 180), self.fine_step_size)
            min_error_fine = np.finfo('f8').max
            optimal_theta_fine = optimal_theta
            for theta in theta_fine_range:
                objective_value = objective_theta(theta)
                if objective_value < min_error_fine:
                    min_error_fine = objective_value
                    optimal_theta_fine = theta

            optimal_thetas[i] = optimal_theta_fine
            print(f"Sample {i}: Optimal theta: {optimal_theta_fine}")

        return optimal_thetas


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    N = int(config['DATA_GENERATION']['N'])
    DATA_FILE = config['DATA_GENERATION']['DATA_FILE']

    optimizer = OrientationOptimizer(N, DATA_FILE)
    optimal_thetas = optimizer.optimize_orientation()
    print("Optimal Thetas:", optimal_thetas)

