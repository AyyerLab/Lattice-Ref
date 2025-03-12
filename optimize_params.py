import h5py
import argparse
import cupy as cp
from utils import get_vals
from utils_cupy import rotate_arr
from configparser import ConfigParser

class ParamOptimizer:
    def __init__(self, niter, N, ftobj, data_file, angles):
        self.N = N
        self.cen = self.N // 2
        self.ITER = niter
        self.data_file = data_file
        self.ftobj = cp.asarray(ftobj)
        self.load_dataset()
        self.angles = cp.asarray(angles)

    def load_dataset(self):
        with h5py.File(self.data_file, 'r') as f:
            self.intens_vals = cp.asarray(f['intens'])
            self.funitc = cp.asarray(f['funitc'])

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

        ftobj_vals = get_vals(rotate_arr(self.N, self.ftobj, angle), self.cen, qh, qk)

        ncoarse = 30
        dx_range = cp.linspace(0, 1, ncoarse)
        dy_range = cp.linspace(0, 1, ncoarse)
        fluence_range = cp.linspace(0, 1, ncoarse)

        optimal_params, min_error = self.grid_search(qh, qk,
                                                     funitc_vals, ftobj_vals, intens_vals,
                                                     dx_range, dy_range, fluence_range)
        dx0, dy0, fluence0 = optimal_params

        gsize = 0.05
        threshold = 1e-5
        for itr in range(2000):
            dx_range = cp.linspace(max(dx0 - 2*gsize, 0), min(dx0 + 2*gsize, 1), 10)
            dy_range = cp.linspace(max(dy0 - 2*gsize, 0), min(dy0 + 2*gsize, 1), 10)
            fluence_range = cp.linspace(max(fluence0 - 2*gsize, 0), min(fluence0 + 2*gsize, 1), 10)

            new_params, new_error = self.grid_search(qh, qk,
                                                     funitc_vals, ftobj_vals, intens_vals,
                                                     dx_range, dy_range, fluence_range)

            if cp.abs(min_error - new_error) < threshold:
                break

            dx0, dy0, fluence0 = new_params
            min_error = new_error
            gsize /= 2

        return (dx0, dy0, fluence0), float(min_error), itr

    def optimize_params(self):
        N = 101
        cen = N // 2
        qh, qk = cp.indices((N, N))
        qh -= cen
        qk -= self.cen
        hk = cp.array([(h, k) for h, k in zip(qh.flatten(), qk.flatten()) if h**2 + k**2 < 20**2])
        num_frames = self.intens_vals.shape[0]

        results = []
        for frame_idx in range(num_frames):
            intens = self.intens_vals[frame_idx]
            angle = self.angles[frame_idx]
            optimal_params, error, itr = self.analyze_frame(intens, hk, angle)

            dx, dy, fluence = optimal_params
            results.append({
                'dx': dx % 1,
                'dy': dy % 1,
                'fluence': fluence,
                'error': error,
                'iter': itr
            })

            print(
                f"ITER {self.ITER}: FRAME {frame_idx + 1}/{num_frames}:"
                f" Dx={dx:.3f}, Dy={dy:.3f},"
                f" Fluence={fluence:.3f}",
                flush=True)

        fitted_dx = cp.array([res['dx'] for res in results])
        fitted_dy = cp.array([res['dy'] for res in results])
        fitted_fluence = cp.array([res['fluence'] for res in results])
        min_errors = cp.array([res['error'] for res in results])
        itrs = cp.array([res['iter'] for res in results])

        return fitted_dx, fitted_dy, fitted_fluence, min_errors, itrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize parameters (shifts, fluence) using ParamOptimizer.")
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    niter = config.getint("OPTIMIZATION", "num_iteration")
    N = config.getint("PARAMETERS", "N")
    data_file = config["FILES"]["data_file"]
    with h5py.File(data_file, "r") as f:
        ftobj = cp.asarray(f["ftobj"][:])
        angles = cp.asarray(f["angles"][:])

    seed = config.getint("PARAMETERS", "seed")
    num_frames = config.getint("PARAMETERS", "num_frames")

    optimizer = ParamOptimizer(niter, N, ftobj, data_file, angles)
    fitted_dx, fitted_dy, fitted_fluence, min_errors, itrs = optimizer.optimize_params()
    with h5py.File('/scratch/mallabhi/lattice_ref/output/optimize_params.h5', "w") as f:
        f['fitted_dx'] = fitted_dx.get()
        f['fitted_dy'] = fitted_dy.get()
        f['fitted_fl'] = fitted_fluence.get()
    print("Optimization Complete.")

