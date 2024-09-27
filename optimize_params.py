import cupy as cp
import h5py
import sys
from cupyx.scipy.ndimage import map_coordinates  # Use cupyx for efficient GPU processing
from utils import get_vals

class ParamOptimizer:
    def __init__(self, N, DATA_FILE, ftobj, INIT_ITER):
        self.N = N
        self.cen = N // 2
        self.DATA_FILE = DATA_FILE
        self.INIT_ITER = INIT_ITER
        self.ftobj = cp.asarray(ftobj)
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])
            self.angles = cp.asarray(f['angle'])  # Load the angles from the dataset

    def rotate_ft(self, ftobj, angle_deg):
        """ Rotate the Fourier transform object in cupy using cupyx map_coordinates """
        angle_rad = cp.deg2rad(angle_deg)
        qh, qk = cp.indices((self.N, self.N))
        qh = qh - self.cen
        qk = qk - self.cen
        qh_rot = cp.cos(angle_rad) * qh - cp.sin(angle_rad) * qk
        qk_rot = cp.sin(angle_rad) * qh + cp.cos(angle_rad) * qk
        coords = cp.array([qh_rot + self.cen, qk_rot + self.cen])
        
        # Use cupyx's map_coordinates for GPU-accelerated interpolation
        rotated_ft = map_coordinates(cp.abs(ftobj), coords, order=3, mode='wrap')
        return rotated_ft

    def grid_search(self, qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range):
        dx_grid, dy_grid, fluence_grid = cp.meshgrid(dx_range, dy_range, fluence_range, indexing='ij')
        dx_grid = dx_grid.ravel()
        dy_grid = dy_grid.ravel()
        fluence_grid = fluence_grid.ravel()

        phase_grid = 2 * cp.pi * (qh[:, None] * dx_grid + qk[:, None] * dy_grid)
        pramp_grid = cp.exp(1j * phase_grid)

        model_intensity = cp.abs(funitc_vals[:, None] + fluence_grid * ftobj_vals[:, None] * pramp_grid) ** 2
        error = cp.sum((model_intensity - intens_vals[:, None]) ** 2, axis=0)

        min_idx = cp.argmin(error)
        optimal_params = dx_grid[min_idx], dy_grid[min_idx], fluence_grid[min_idx]
        min_error = error[min_idx]

        return optimal_params, min_error

    def analyze_frame(self, intens, hk, angle):
        hk = cp.array(hk)
        qh, qk = hk[:, 0], hk[:, 1]

        funitc_vals = get_vals(self.funitc, self.cen, qh, qk)
        intens_vals = get_vals(intens, self.cen, qh, qk)

        # Initial coarse grid search
        ncoarse = 300
        dx_range = cp.linspace(0, 1, ncoarse)
        dy_range = cp.linspace(0, 1, ncoarse)
        fluence_range = cp.linspace(0.1, 10, ncoarse)

        # Rotate ftobj using the provided angle for this frame
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
            dx_range = cp.linspace(max(dx0 - gsize, 0), min(dx0 + gsize, 1), 10)
            dy_range = cp.linspace(max(dy0 - gsize, 0), min(dy0 + gsize, 1), 10)
            fluence_range = cp.linspace(max(fluence0 - 2 * gsize, 0.1), min(fluence0 + 2 * gsize, 10), 10)

            new_params, new_error = self.grid_search(
                qh, qk, funitc_vals, ftobj_vals, intens_vals, dx_range, dy_range, fluence_range
            )

            if cp.abs(min_error - new_error) < threshold:
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
            angle = self.angles[frame_idx]  # Use the loaded angle for each frame
            optimal_params, error = self.analyze_frame(intens, hk, angle)
            dx, dy, fluence = optimal_params
            results.append({
                'dx': dx % 1,
                'dy': dy % 1,
                'fluence': fluence,
                'error': error
            })
            print(
                f"ITER {self.INIT_ITER}: FRAME {frame_idx + 1}/{num_frames}: "
                f"Dx={dx:.3f}, Dy={dy:.3f}, Fluence={fluence:.3f}, Error={error:.3e}, Angle={angle:.2f}",
                flush=True
            )

        fitted_dx = cp.array([res['dx'] for res in results])
        fitted_dy = cp.array([res['dy'] for res in results])
        fitted_fluence = cp.array([res['fluence'] for res in results])
        min_errors = cp.array([res['error'] for res in results])

        return fitted_dx, fitted_dy, fitted_fluence, min_errors

