import numpy as np
import h5py
from scipy import ndimage
from scipy.optimize import curve_fit
import configparser

class Align:
    def __init__(self, fitted_ftobj_path, ftobj_path, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.N = int(config['DATA_GENERATION']['N'])
        self.PIXELS = int(config['OPTIMIZATION']['PIXELS'])
        self.SIGMA = float(config['ALIGN']['SIGMA'])
        self.REGION = tuple(map(int, config['ALIGN']['REGION'].split(',')))

        # Load data
        with h5py.File(fitted_ftobj_path, 'r') as f:
            self.fitted_ftobj = f['fitted_ftobj'][:]
        with h5py.File(ftobj_path, 'r') as f:
            self.ftobj = f['ftobj'][:]

    def do_fft(self, obj):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

    def do_ifft(self, ftobj):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

    def _get_supp(self, ftobj, sig, pixels):
        invsuppmask = np.ones((self.N,) * 2, dtype=np.bool_)
        amodel = np.real(self.do_ifft(ftobj.reshape((self.N,) * 2)))
        samodel = ndimage.gaussian_filter(amodel, sig)
        thresh = np.quantile(samodel, (samodel.size - pixels) / samodel.size)
        invsuppmask = samodel < thresh
        return invsuppmask

    def center_obj(self, obj, clean_obj):
        x, y = np.indices(clean_obj.shape)
        cx = (np.abs(clean_obj) * x).sum() / np.abs(clean_obj).sum()
        cy = (np.abs(clean_obj) * y).sum() / np.abs(clean_obj).sum()
        obj_cen = np.roll(np.abs(obj), (self.N//2 - int(cx), self.N//2 - int(cy)), axis=(0, 1))
        return obj_cen

    def fit_pramp(self, pramp):
        def linear_fun(coords, A, B, C):
            x, y = coords
            return A * x + B * y + C

        h, w = pramp.shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        coords = np.vstack((X.flatten(), Y.flatten()))
        z_flat = pramp.flatten()
        params, _ = curve_fit(linear_fun, coords, z_flat)
        return params

    def fix_phase(self, fobj, pramp, region=None):
        pramp_region = pramp if region is None else pramp[region[0]:region[1], region[2]:region[3]]
        params = self.fit_pramp(pramp_region)
        A, B, C = params
        h, w = pramp.shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        fit_pramp_vals = (A * X + B * Y + C)
        fobj_align = fobj * np.exp(1j * fit_pramp_vals)
        return fobj_align

    def align(self):
        tobj = np.abs(self.do_ifft(self.ftobj))
        fitted_tobj = np.abs(self.do_ifft(self.fitted_ftobj))

        fitted_tobj_copy = fitted_tobj.copy()
        support = self._get_supp(self.fitted_ftobj, self.SIGMA, self.PIXELS)
        fitted_tobj_copy[support] = 0

        fitted_tobj_cen = self.center_obj(fitted_tobj, fitted_tobj_copy)
        tobj_cen = self.center_obj(tobj, tobj)

        pramp = np.angle(self.do_fft(tobj_cen) / self.do_fft(fitted_tobj_cen))
        fitted_ftobj_align = self.fix_phase(self.do_fft(fitted_tobj_cen), pramp, self.REGION)
        return fitted_ftobj_align, self.do_fft(tobj_cen)



# IPYTHON EXECUTION
#ftobj_path = '/path/to/dataset.h5'
#fitted_ftobj_path = '/path/to/fitted_dataset.h5'
#aligner = Align(ftobj_path, fitted_ftobj_path)
#aligned_fitted_ftobj, aligned_ftobj = aligner.align()

