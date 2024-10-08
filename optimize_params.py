import numpy as np
import h5py
import sys
import configparser
from scipy.ndimage import map_coordinates
from utils import get_vals

class ParamOptimizer:
    def __init__(self, N, DATA_FILE):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.funitc = f['funitc'][:]
            self.angles = f['angle'][:]
            self.ftobj = f['ftobj'][:]
        print(f"Loaded {self.intens_vals.shape[0]} frames", flush=True)

    def rotate_ft(self, ftobj, angle_deg):
        """ Rotate the Fourier transform object in numpy using scipy map_coordinates """
        angle_rad = np.deg2rad(angle_deg)
        qh, qk = np.indices((self.N, self.N))
        qh = qh - self.cen
        qk = qk - self.cen
        qh_rot = np.cos(angle_rad) * qh - np.sin(angle_rad) * qk
        qk_rot = np.sin(angle_rad) * qh + np.cos(angle_rad) * qk
        coords = np.array([qh_rot + self.cen, qk_rot + self.cen])
        rotated_ft = map_coordinates(np.abs(ftobj), coords, order=3, mode='wrap')
        return rotated_ft

    def grid_search(self, qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range):
        dx_grid, dy_grid, fluence_grid = np.meshgrid(dx_range, dy_range, fluence_range, indexing='ij')
        dx_grid = dx_grid.ravel()
        dy_grid = dy_grid.ravel()
        fluence_grid = fluence_grid.ravel()

        phase_grid = 2 * np.pi * (qh[:, None] * dx_grid + qk[:, None] * dy_grid)
        pramp_grid = np.exp(1j * phase_grid)

        model_intensity = np.abs(funitc_vals[:, None] + fluence_grid * ftobj_vals[:, None] * pramp_grid) ** 2
        error = np.sum((model_intensity - intens_vals[:, None]) ** 2, axis=0)

        min_idx = np.argmin(error)
        optimal_params = dx_grid[min_idx], dy_grid[min_idx], fluence_grid[min_idx]
        min_error = error[min_idx]

        return optimal_params, min_error

    def analyze_frame(self, intens, hk, angle):
        hk = np.array(hk)
        qh, qk = hk[:, 0], hk[:, 1]

        funitc_vals = get_vals(self.funitc, self.cen, qh, qk)
        intens_vals = get_vals(intens, self.cen, qh, qk)

        # Initial coarse grid search
        ncoarse = 300
        dx_range = np.linspace(0, 1, ncoarse)
        dy_range = np.linspace(0, 1, ncoarse)
        fluence_range = np.linspace(0.1, 10, ncoarse)

        rotated_ftobj = self.rotate_ft(self.ftobj, angle)
        ftobj_vals = get_vals(rotated_ftobj, self.cen, qh, qk)

        optimal_params, min_error = self.grid_search(
            qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range
        )

        # Refinement loop
        gsize = 0.05
        threshold = 1e-4
        for _ in range(1000):
            dx0, dy0, fluence0 = optimal_params
            dx_range = np.linspace(max(dx0 - gsize, 0), min(dx0 + gsize, 1), 10)
            dy_range = np.linspace(max(dy0 - gsize, 0), min(dy0 + gsize, 1), 10)
            fluence_range = np.linspace(max(fluence0 - 2 * gsize, 0.1), min(fluence0 + 2 * gsize, 10), 10)

            new_params, new_error = self.grid_search(
                qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range
            )

            if np.abs(min_error - new_error) < threshold:
                break

            optimal_params, min_error = new_params, new_error
            gsize /= 2

        return tuple(map(float, optimal_params)), float(min_error)

    def optimize_params(self):
        hk = [(0, 1), (1, 1), (1, 0), (1, -1)]
        num_frames = self.intens_vals.shape[0]

        results = []
        for frame_idx in range(num_frames):
            intens = self.intens_vals[frame_idx]
            angle = self.angles[frame_idx]
            optimal_params, error = self.analyze_frame(intens, hk, angle)
            dx, dy, fluence = optimal_params
            results.append({
                'dx': dx % 1,
                'dy': dy % 1,
                'fluence': fluence,
                'error': error
            })
            print(
                f"FRAME {frame_idx + 1}/{num_frames}: "
                f"Dx={dx:.3f}, Dy={dy:.3f}, Fluence={fluence:.3f}, Error={error:.3e}, Angle={angle:.2f}",
                flush=True
            )

        fitted_dx = np.array([res['dx'] for res in results])
        fitted_dy = np.array([res['dy'] for res in results])
        fitted_fluence = np.array([res['fluence'] for res in results])
        min_errors = np.array([res['error'] for res in results])

        return fitted_dx, fitted_dy, fitted_fluence, min_errors


def load_config(config_file):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    N = config.getint('DATA_GENERATION', 'N')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')

    return N, DATA_FILE


if __name__ == "__main__":
    config_file = 'config.ini'
    N, DATA_FILE = load_config(config_file)
    try:
        optimizer = ParamOptimizer(N, DATA_FILE)
        fitted_dx, fitted_dy, fitted_fluence, min_errors = optimizer.optimize_params()
        print("Optimization complete.", flush=True)
        print(f"Fitted Dx: {fitted_dx}", flush=True)
        print(f"Fitted Dy: {fitted_dy}", flush=True)
        print(f"Fitted Fluence: {fitted_fluence}", flush=True)
        print(f"Min Errors: {min_errors}", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        sys.exit(1)

