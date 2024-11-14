import numpy as np
import h5py

from scipy import ndimage
from scipy.optimize import curve_fit
import configparser

from align_obj import Align

class FRC:
    def __init__(self, fitted_ftobj_path, ftobj_path, config_path):

        self.ALIGN_OBJ = Align(fitted_ftobj_path, 
                               ftobj_path, 
                               config_path)

    def calculate_frc(self, obj1, obj2, binsize=1.0, do_abs=False):
        fobj1 = self.ALIGN_OBJ.do_fft(obj1)
        fobj2 = self.ALIGN_OBJ.do_fft(obj2)

        x, y = np.meshgrid(np.arange(self.ALIGN_OBJ.N) - self.ALIGN_OBJ.N // 2,
                           np.arange(self.ALIGN_OBJ.N) - self.ALIGN_OBJ.N // 2, 
                           indexing='ij')

        binrad = (np.sqrt(x**2 + y**2) / binsize).astype(int)
        rsize = binrad.max() + 1

        numr = np.zeros(rsize, dtype=np.complex128)
        denr1 = np.zeros(rsize, dtype=np.float64)
        denr2 = np.zeros(rsize, dtype=np.float64)

        np.add.at(numr, binrad, fobj1 * np.conj(fobj2))
        np.add.at(denr1, binrad, np.abs(fobj1)**2)
        np.add.at(denr2, binrad, np.abs(fobj2)**2)

        denr = np.sqrt(denr1 * denr2)
        frc_vals = np.zeros(rsize, dtype=np.float64)
        valid = denr > 0

        if do_abs:
            frc_vals[valid] = np.abs(numr[valid]) / denr[valid]
        else:
            frc_vals[valid] = np.real(numr[valid]) / denr[valid]
        rvals = np.arange(rsize) * binsize
        return rvals, frc_vals

    def process(self):
        aligned_fitted_ftobj, aligned_ftobj = self.ALIGN_OBJ.align()
        rvals, frc_vals = self.calculate_frc(self.ALIGN_OBJ.do_ifft(aligned_fitted_ftobj),
                                             self.ALIGN_OBJ.do_ifft(aligned_ftobj))
        return rvals, frc_vals

