import h5py
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from utils import do_fft, do_ifft

class FRC:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
        self.N = 127

    def load_data(self):
        with h5py.File(self.file1, "r") as f:
            self.ftobj_pred = f['ftobj_fitted'][:]
        with h5py.File(self.file2, "r") as f:
            self.ftobj = f['ftobj'][:]
            self.tobj = f['tobj'][:]
    
    def do_ifft(self, obj):
        return do_ifft(obj)
    
    def _get_supp(self, ftobj, pixs=1000):
        amodel = np.abs(self.do_ifft(ftobj.reshape((self.N,)*2)))
        famodel = ndimage.gaussian_filter(amodel, 3)
        thresh = np.sort(famodel.ravel())[amodel.size - pixs]
        supp = famodel < thresh
        return supp
    
    def calc_com(self, obj):
        x, y = np.indices(obj.shape) 
        cx = (np.abs(obj) * x).sum() / np.abs(obj).sum()
        cy = (np.abs(obj) * y).sum() / np.abs(obj).sum()
        return cx, cy

    def center_obj(self, obj):
        cx, cy = self.calc_com(obj)
        obj_cen = np.roll(np.abs(obj), (self.N//2 - int(cx), self.N//2 - int(cy)), axis=(0, 1))
        ft_obj_cen = do_fft(obj_cen)
        return obj_cen, ft_obj_cen
    
    def _get_pramp(self, obj1, obj2):
        obj1_cen, ftobj1_cen = self.center_obj(obj1)
        obj2_cen, ftobj2_cen = self.center_obj(obj2)
        pramp = np.angle(ftobj1_cen / ftobj2_cen)
        return pramp, obj1_cen, obj2_cen, ftobj1_cen, ftobj2_cen

    def fit_pramp(self, pramp):
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
    
    def fix_phase(self, obj_pred, pramp):
        params = self.fit_pramp(pramp[25:95, 25:95])
        A, B, C = params

        h, w = pramp.shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)

        x_flat = X.flatten()
        y_flat = Y.flatten()
        fit_vals = (A * x_flat + B * y_flat + C).reshape(pramp.shape)
        fit_pramp = fit_vals

        ftobj_pred_cen = do_fft(obj_pred)
        ftobj_pred_fix = ftobj_pred_cen * np.exp(-1j * fit_pramp)
        obj_pred_fix = do_ifft(ftobj_pred_fix)
        return obj_pred_fix
    
    def calculate_frc(self, obj1, obj2, binsize=1., do_abs=False):
        fobj1 = do_fft(obj1)
        fobj2 = do_fft(obj2)
        x, y = np.meshgrid(np.arange(self.N) - self.N // 2, np.arange(self.N) - obj1.shape[1] // 2, indexing='ij')
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
        self.load_data()
        obj_pred = self.do_ifft(self.ftobj_pred)
        supp = self._get_supp(self.ftobj_pred)
        obj_pred[supp] = 0
        
        pramp, obj_pred_cen, tobj_cen, ftobj_pred_cen, ftobj_cen = self._get_pramp(obj_pred, self.tobj)
        obj_pred_fix = self.fix_phase(obj_pred_cen, pramp)
        rvals, frc_vals = self.calculate_frc(tobj_cen, obj_pred_fix)
        return rvals, frc_vals



#frc = FRC('/u/mallabhi/lattice_ref/data/output/output150.h5',
#                            '/u/mallabhi/lattice_ref/data/dataset.h5')

#rvals, frc_vals = frc.process()

