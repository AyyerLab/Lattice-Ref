import numpy as np
import h5py
import sys
from utils import load_config, get_vals
from optimize_params import Optimizer

class ConjugateGradientOptimizer:
    def __init__(self, N, DATA_FILE, SCALE, shifts, fluence_vals, OUTPUT_FILE):
        self.N = N
        self.cen = self.N // 2
        self.SCALE = SCALE
        self.DATA_FILE = DATA_FILE
        self.OUTPUT_FILE = OUTPUT_FILE

        self.shifts = shifts
        self.fluence_vals = fluence_vals
        self.load_dataset()

    def load_dataset(self):
        with h5py.File(self.DATA_FILE, 'r') as f:
            self.intens_vals = f['intens'][:]
            self.funitc = f['funitc'][:]
            self.num_samples = self.intens_vals.shape[0]

    def optimize_pixel(self, qh, qk):
        funitc_pixvals = [get_vals(self.funitc, self.cen, qh, qk)]
        intens_pixvals = np.array([get_vals(self.intens_vals[i], self.cen, qh, qk) for i in range(self.num_samples)])

        def pix_objective(params):
            ftobj_real, ftobj_imag = params
            ftobj_val = ftobj_real + 1j * ftobj_imag
            error = 0
            funitc_val = funitc_pixvals[0]

            for d in range(self.num_samples):
                phase = 2.0 * np.pi * (qh * self.shifts[d, 0] + qk * self.shifts[d, 1])
                pramp = np.exp(1j * phase)
                model_int = self.fluence_vals[d] * np.abs(self.SCALE * funitc_val + ftobj_val * pramp)**2
                error += (model_int - intens_pixvals[d])**2

            return error

        def pix_objective_grad(params):
            ftobj_real, ftobj_imag = params
            ftobj_val = ftobj_real + 1j * ftobj_imag
            grad_real = 0
            grad_imag = 0
            funitc_val = funitc_pixvals[0]

            for d in range(self.num_samples):
                phase = 2.0 * np.pi * (qh * self.shifts[d, 0] + qk * self.shifts[d, 1])
                pramp = np.exp(1j * phase)
                F = self.SCALE * funitc_val + ftobj_val * pramp
                model_int = self.fluence_vals[d] * np.abs(F)**2
                residual = model_int - intens_pixvals[d]

                F_real = self.SCALE * funitc_val.real + ftobj_real * np.cos(phase) - ftobj_imag * np.sin(phase)
                F_imag = self.SCALE * funitc_val.imag + ftobj_real * np.sin(phase) + ftobj_imag * np.cos(phase)

                d_model_int_d_real = 2 * self.fluence_vals[d] * (F_real * np.cos(phase) + F_imag * np.sin(phase))
                d_model_int_d_imag = 2 * self.fluence_vals[d] * (F_imag * np.cos(phase) - F_real * np.sin(phase))

                grad_real += 2 * residual * d_model_int_d_real
                grad_imag += 2 * residual * d_model_int_d_imag

            return np.array([grad_real, grad_imag])

        def line_search(f, grad, params, direction, alpha=1.0, beta=0.5, sigma=1e-4):
            while f(params + alpha * direction) > f(params) + sigma * alpha * np.dot(grad, direction):
                alpha *= beta
            return alpha

        def conjugate_gradient_descent(f, grad_f, initial_params, tol=1e-6, max_iter=1000):
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

        for h in range(-self.cen, self.cen):
            for k in range(-self.cen, self.cen):
                params = self.optimize_pixel(h, k)
                optimized_params[h + self.cen, k + self.cen] = params
                pixel_count += 1
                self._print_progress(pixel_count, total_pixels)

        self.save_results(optimized_params)
        return optimized_params


    def _print_progress(self, count, total):
        progress = (count / total) * 100
        sys.stdout.write(f'\rOptimized {count}/{total} pixels ({progress:.2f}%)')
        sys.stdout.flush()

    def save_results(self, optimized_params):
        with h5py.File(self.OUTPUT_FILE, 'w') as f:
            f.create_dataset('fitted_dx', data=self.shifts[:, 0])
            f.create_dataset('fitted_dy', data=self.shifts[:, 1])
            f.create_dataset('fitted_fluence', data=self.fluence_vals)
            f.create_dataset('ftobj_fitted', data=optimized_params)

