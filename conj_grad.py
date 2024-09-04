import numpy as np
import h5py
import sys
from utils import get_vals

class ConjugateGradientOptimizer:
    def __init__(self, N, DATA_FILE, SCALE, shifts, fluence_vals, OUTPUT_FILE, itern):
        self.N = N
        self.cen = self.N // 2
        self.SCALE = SCALE
        self.ITERATION = itern
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE

        self.shifts = shifts
        self.fluence_vals = fluence_vals
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.funitc = f['funitc'][:]
            self.NUM_SAMPLES = self.intens_vals.shape[0]

    def optimize_pixel(self, qh, qk):
        funitc_pixvals = get_vals(self.funitc, self.cen, qh, qk)
        intens_pixvals = np.array([get_vals(self.intens_vals[i], self.cen, qh, qk) for i in range(self.NUM_SAMPLES)])

        phases = 2.0 * np.pi * (qh * self.shifts[:, 0] + qk * self.shifts[:, 1])
        pramp = np.exp(1j * phases)

        def pix_objective(params):
            ftobj_real, ftobj_imag = params
            ftobj_val = ftobj_real + 1j * ftobj_imag
            funitc_val = funitc_pixvals

            F = funitc_val + self.fluence_vals * ftobj_val * pramp
            model_int = np.abs(F)**2
            residuals = model_int - intens_pixvals
            error = np.sum(residuals**2)
            return error

        def pix_objective_grad(params):
            ftobj_real, ftobj_imag = params
            ftobj_val = ftobj_real + 1j * ftobj_imag
            funitc_val = funitc_pixvals

            F = funitc_val + self.fluence_vals * ftobj_val * pramp
            model_int = np.abs(F)**2
            residuals = model_int - intens_pixvals

            F_real = F.real
            F_imag = F.imag

            dF_dreal = self.fluence_vals * pramp.real
            dF_dimag = self.fluence_vals * pramp.imag

            d_model_int_d_real = 2 * (F_real * dF_dreal + F_imag * dF_dimag)
            d_model_int_d_imag = 2 * (F_imag * dF_dreal - F_real * dF_dimag)

            grad_real = 2 * np.sum(residuals * d_model_int_d_real)
            grad_imag = 2 * np.sum(residuals * d_model_int_d_imag)

            return np.array([grad_real, grad_imag])

        def line_search(f, grad, params, direction, alpha=0.8, beta=0.4, sigma=85e-2):
            while f(params + alpha * direction) > f(params) + sigma * alpha * np.dot(grad, direction):
                alpha *= beta
            return alpha

        def conjugate_gradient_descent(f, grad_f, initial_params, tol=1e-8, max_iter=2000):
            params = initial_params
            grad = grad_f(params)
            direction = -grad
            for i in range(max_iter):
                alpha = line_search(f, grad, params, direction)
                new_params = params + alpha * direction
                new_grad = grad_f(new_params)
                beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
                new_direction = -new_grad + beta * direction

                if np.linalg.norm(new_params - params) < tol or np.linalg.norm(new_grad) < tol:
                    break

                params = new_params
                grad = new_grad
                direction = new_direction

            return params

        initial_params = np.array([0, 0])
        optimized_params = conjugate_gradient_descent(pix_objective, pix_objective_grad, initial_params)
        return optimized_params

    def optimize_all_pixels(self):
        optimized_params = np.zeros((self.N, self.N, 2))
        total_pixels = (self.N * self.N)
        pixel_count = 0

        for h in range(-self.cen, self.cen + 1):
            for k in range(-self.cen, self.cen + 1):
                params = self.optimize_pixel(h, k)
                optimized_params[h + self.cen, k + self.cen] = params
                pixel_count += 1
                self._print_progress(pixel_count, total_pixels)

        self.save_results(optimized_params, self.ITERATION)
        return optimized_params

    def _print_progress(self, count, total):
        PROGRESS = (count / total) * 100
        sys.stdout.write(f'\rOPTIMIZED {count}/{total} PIXELS ({PROGRESS:.2f}%)')
        sys.stdout.flush()

    def save_results(self, optimized_params, iteration):
        output_file = self.OUTPUT_FILE.replace('.h5', f'{iteration}.h5')

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('fitted_dx', data=self.shifts[:, 0])
            f.create_dataset('fitted_dy', data=self.shifts[:, 1])
            f.create_dataset('fitted_fluence', data=self.fluence_vals)
            f.create_dataset('ftobj_fitted', data=optimized_params)

