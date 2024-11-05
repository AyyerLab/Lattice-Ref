import numpy as np
import h5py
from scipy import ndimage
from scipy.optimize import curve_fit

class Align:
    def __init__(self, ftobj_path, fitted_ftobj_path):
        self.N = 127
        with h5py.File(ftobj_path, 'r') as f:
            self.ftobj = f['ftobj'][:]
        with h5py.File(fitted_ftobj_path, 'r') as f:
            self.fitted_ftobj = f['fitted_ftobj'][:]
    
    def do_fft(self, obj):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

    def do_ifft(self, ftobj):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

    def _get_supp(self, ftobj, sig, pixels):
        # Get estimate of support
        invsuppmask = np.ones((self.N,) * 2, dtype=np.bool_)
        amodel = np.real(self.do_ifft(ftobj.reshape((self.N,) * 2)))
        samodel = ndimage.gaussian_filter(amodel, sig)
        thresh = np.quantile(samodel, (samodel.size - pixels) / samodel.size)
        invsuppmask = samodel < thresh
        return invsuppmask

    def center_obj(self, obj, clean_obj):
        # Center the object based on the centroid of clean_obj
        x, y = np.indices(clean_obj.shape)
        cx = (np.abs(clean_obj) * x).sum() / np.abs(clean_obj).sum()
        cy = (np.abs(clean_obj) * y).sum() / np.abs(clean_obj).sum()
        obj_cen = np.roll(np.abs(obj), (self.N//2 - int(cx), self.N//2 - int(cy)), axis=(0, 1))
        return obj_cen

    def fit_pramp(self, pramp):
        # Fit lunear function to phase ramp
        def linear_fun(coords, A, B, C):
            x, y = coords
            return A * x + B * y + C
        h, w = pramp.shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = pramp.flatten()
        coords = np.vstack((x_flat, y_flat))
        params, _ = curve_fit(linear_fun, coords, z_flat)
        return params

    def fix_phase(self, fobj, pramp, region=None):
        # Use a specific region for fitting if provided
        if region is not None:
            y1, y2, x1, x2 = region
            pramp_region = pramp[y1:y2, x1:x2]
        else:
            pramp_region = pramp
        params = self.fit_pramp(pramp_region)
        A, B, C = params
        h, w = pramp.shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        fit_pramp_vals = (A * X + B * Y + C)
        # Apply phase ramp
        fobj_align = fobj * np.exp(1j * fit_pramp_vals)
        return fobj_align

    def align(self, sig=1.25, pixels=830, region=(40, 85, 45, 80)):
        tobj = np.abs(self.do_ifft(self.ftobj))
        fitted_tobj = np.abs(self.do_ifft(self.fitted_ftobj))

        # Compute support mask
        fitted_tobj_copy = fitted_tobj.copy()
        support = self._get_supp(self.fitted_ftobj, sig, pixels)
        fitted_tobj_copy[support] = 0
        # Center the objects
        fitted_tobj_cen = self.center_obj(fitted_tobj, fitted_tobj_copy)
        tobj_cen = self.center_obj(tobj, tobj)
        # Compute pramp
        pramp = np.angle(self.do_fft(tobj_cen) / self.do_fft(fitted_tobj_cen))
        # Fix the phase
        fitted_ftobj_align = self.fix_phase(self.do_fft(fitted_tobj_cen), pramp, region)
        return fitted_ftobj_align, self.do_fft(tobj_cen)

####  Execution

#ftobj_path = '/scratch/mallabhi/lattice_ref/data/K/dataset_K10.h5'
#fitted_ftobj_path = '/scratch/mallabhi/lattice_ref/output/output_K10_NS_150.h5'
#aligner = ao.Align(ftobj_path, fitted_ftobj_path)
#aligned_fitted_ftobj, aligned_ftobj = aligner.align()



