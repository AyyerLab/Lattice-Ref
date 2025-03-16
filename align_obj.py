import os
import h5py
import argparse
import numpy as np
import configparser
from scipy import ndimage

from utils import do_fft, do_ifft
from scipy.optimize import curve_fit
from calc_frc import FRC

class Align:
    def __init__(self, fitted_ftobj, ftobj, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        self.N = int(config['PARAMETERS']['N'])
        self.PIXELS = int(config['OPTIMIZATION']['PIXELS'])
        self.SIGMA = float(config['ALIGN']['SIGMA'])
        self.REGION = tuple(map(int, config['ALIGN']['REGION'].split(',')))

        self.fitted_ftobj = self._get_ftobjs(fitted_ftobj, 'fitted_ftobj')
        self.ftobj = self._get_ftobjs(ftobj, 'ftobj')

    def _get_ftobjs(self, input_ftobj, dataset_name):
        if isinstance(input_ftobj, str) and os.path.exists(input_ftobj):
            with h5py.File(input_ftobj, 'r') as f:
                return f[dataset_name][:]
        elif isinstance(input_ftobj, np.ndarray):
            return input_ftobj
        else:
            raise ValueError(f"{dataset_name} input must be a file path (str) or a NumPy array.")

    def _get_supp(self, ftobj, sig, pixels):
        amodel = np.real(do_ifft(ftobj.reshape((self.N,) * 2)))
        samodel = ndimage.gaussian_filter(amodel, sig)
        thresh = np.quantile(samodel, (samodel.size - pixels) / samodel.size)
        invsuppmask = samodel < thresh
        return invsuppmask

    def get_com(self, obj):
        x, y = np.indices(obj.shape)
        cx = (np.abs(obj) * x).sum() / np.abs(obj).sum()
        cy = (np.abs(obj) * y).sum() / np.abs(obj).sum()
        return cx, cy

    def center_obj(self, obj, clean_obj):
        cx, cy = self.get_com(clean_obj)
        shift = (self.N // 2 - int(cx), self.N // 2 - int(cy))
        obj_cen = np.roll(np.abs(obj), shift, axis=(0, 1))
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
        fit_pramp_vals = A * X + B * Y + C
        ftobj_align = fobj * np.exp(1j * fit_pramp_vals)
        return ftobj_align

    def align(self):
        tobj = np.abs(do_ifft(self.ftobj))
        fitted_tobj = np.abs(do_ifft(self.fitted_ftobj))

        fitted_tobj_copy = fitted_tobj.copy()
        support = self._get_supp(self.fitted_ftobj, self.SIGMA, self.PIXELS)
        fitted_tobj_copy[support] = 0

        # Centering the objects
        fitted_tobj_cen = self.center_obj(fitted_tobj, fitted_tobj_copy)
        tobj_cen = self.center_obj(tobj, tobj)

        # Rotation
        frc = FRC(obj1=tobj_cen, obj2=fitted_tobj_cen, verbose=True)
        _, _, best_ang = frc.calc_rot(binsize=1.0, num_rot=1800, do_abs=False)

        rotfmodel = np.empty_like(do_fft(fitted_tobj_cen))
        rotfmodel.real = ndimage.rotate(do_fft(fitted_tobj_cen).real, best_ang, order=1, prefilter=False, reshape=False)
        rotfmodel.imag = ndimage.rotate(do_fft(fitted_tobj_cen).imag, best_ang, order=1, prefilter=False, reshape=False)
        rot_fitted_tobj_cen = do_ifft(rotfmodel)

        #Phase correction
        pramp = np.angle(do_fft(tobj_cen) / do_fft(rot_fitted_tobj_cen))
        fitted_ftobj_aligned = self.fix_phase(do_fft(rot_fitted_tobj_cen), pramp, self.REGION)

        return do_ifft(fitted_ftobj_aligned), tobj_cen, pramp

