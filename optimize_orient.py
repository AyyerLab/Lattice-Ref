import cupy as cp
import h5py
from scipy.ndimage import map_coordinates
from utils import get_vals

class OrientOptimizer:
    def __init__(self, N, DATA_FILE, ftobj, hk):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.ftobj = cp.asarray(ftobj)
        self.hk = hk
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])

        qh, qk = cp.indices((self.N, self.N))
        self.qh = qh - self.cen
        self.qk = qk - self.cen

    def rotate_ft(self, ftobj, angle_deg):
        angle_rad = cp.deg2rad(angle_deg)
        qh_rot = cp.cos(angle_rad) * self.qh - cp.sin(angle_rad) * self.qk
        qk_rot = cp.sin(angle_rad) * self.qh + cp.cos(angle_rad) * self.qk
        coords = cp.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(cp.abs(ftobj).get(), coords.get(), order=3, mode='wrap')
        return cp.asarray(rotated_ft)

    def grid_search_theta(self, qh, qk, dx, dy, fluence, funitc_vals, intens_vals_sample, ftobj_vals):
        # Coarse grid search for theta
        ncoarse = 20
        theta_range = cp.linspace(0, 180, ncoarse)
        min_error = cp.inf
        optimal_theta = 0

        for theta in theta_range:
            error = self.objective(theta, dx, dy, fluence, funitc_vals, intens_vals_sample, ftobj_vals)
            if error < min_error:
                min_error = error
                optimal_theta = theta

        # Fine grid search around the optimal theta
        gsize = 0.01
        theta_fine_range = cp.arange(max(optimal_theta - 5, 0), min(optimal_theta + 5, 180), gsize)
        for theta in theta_fine_range:
            error = self.objective(theta, dx, dy, fluence, funitc_vals, intens_vals_sample, ftobj_vals)
            if error < min_error:
                min_error = error
                optimal_theta = theta

        return optimal_theta, min_error

    def objective(self, theta, dx, dy, fluence, funitc_vals, intens_vals_sample, ftobj_vals):
        rotated_ftobj = self.rotate_ft(ftobj_vals, theta)
        phase_grid = 2 * cp.pi * (self.qh * dx + self.qk * dy)
        pramp = cp.exp(1j * phase_grid)

        model_intensity = cp.abs(funitc_vals + fluence * rotated_ftobj * pramp) ** 2
        error = cp.sum((model_intensity - intens_vals_sample) ** 2)

        return error

    def analyze_frame(self, intens, dx, dy, fluence):
        hk = cp.array(self.hk)
        qh, qk = hk[:, 0], hk[:, 1]

        funitc_vals = get_vals(self.funitc, self.cen, qh, qk)
        ftobj_vals = get_vals(self.ftobj, self.cen, qh, qk)
        intens_vals_sample = get_vals(intens, self.cen, qh, qk)

        optimal_theta, min_error = self.grid_search_theta(qh, qk, dx, dy, fluence, funitc_vals, intens_vals_sample, ftobj_vals)

        return optimal_theta, min_error

    def optimize_orientations(self, dx_vals, dy_vals, fluence_vals):
        num_frames = self.intens_vals.shape[0]

        results = []
        for frame_idx in range(num_frames):
            intens = self.intens_vals[frame_idx]
            dx, dy, fluence = dx_vals[frame_idx], dy_vals[frame_idx], fluence_vals[frame_idx]

            optimal_theta, error = self.analyze_frame(intens, dx, dy, fluence)
            results.append({
                'theta': optimal_theta,
                'error': error
            })
            print(
                f"FRAME {frame_idx + 1}/{num_frames}: "
                f"Theta={optimal_theta:.3f}, Error={error:.3e}",
                flush=True
            )

        fitted_thetas = cp.array([res['theta'] for res in results])
        min_errors = cp.array([res['error'] for res in results])

        return fitted_thetas, min_errors

