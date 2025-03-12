import os
import ast
import h5py
import mrcfile
import numpy as np
import configparser
from scipy import ndimage
from utils import rotate_arr, do_fft, do_ifft
from config_functions import makedata_config

class DataGenerator:
    def __init__(self, n, scale, num_frames, seed, sample, add_noise, data_file, exposure_time, background_sqr):
        self.N = n
        self.SCALE = scale
        self.NUM_FRAMES = num_frames
        self.SEED = seed
        self.SAMPLE = sample
        self.ADD_NOISE = add_noise
        self.DATA_FILE = data_file
        self.EXPOSURE_TIME = exposure_time
        self.BACKGROUND = np.full((self.N, self.N), np.sqrt(background_sqr))

        self.cen = self.N // 2
        self.qh, self.qk = np.indices((self.N, self.N))
        self.qh -= self.cen
        self.qk -= self.cen

        xvals = np.linspace(0, 1, 500)
        self.max_fl = self.pfluence(xvals[1:]).max()
        np.random.seed(self.SEED)
        self.fluence_vals = self.randfluence(self.NUM_FRAMES)

        np.random.seed(self.SEED)
        self.dx_vals = np.random.uniform(0.01, 1, size=self.NUM_FRAMES)
        self.dy_vals = np.random.uniform(0.01, 1, size=self.NUM_FRAMES)
        self.angles = np.random.uniform(0, 1, size=self.NUM_FRAMES) * 2. * np.pi

    def pfluence(self, xvals):
        return np.sqrt(-np.log(xvals)) * (1 - np.exp(-xvals * 200))

    def randfluence(self, nvals):
        naccept = 0
        phivals = np.zeros(nvals)
        while naccept < nvals:
            rand2 = np.random.random(size=(nvals - naccept, 2))
            rand2[:, 0] = np.clip(rand2[:, 0], 1e-10, 1)
            rand2[:, 1] *= self.max_fl
            sel = np.where(rand2[:, 1] < self.pfluence(rand2[:, 0]))[0]
            nsel = sel.size
            phivals[naccept:naccept + nsel] = rand2[sel, 0]
            naccept += nsel
        return phivals

    def target_obj(self):
        size = self.N
        mcen = size // 2
        x, y = np.indices((size, size), dtype='f8')
        mask = np.zeros((size, size), dtype='f8')

        BASE_DATA_PATH = '/scratch/mallabhi/lattice_ref/data'
        file_names = {
            'PS1_5A': 'PS1_map.mrc',
            'PS1_10A': 'PS1_10angs.mrc',
            'RIB_5A': 'rib.mrc',
            'RIB_10A': 'Rib_10angs.mrc'}

        if self.SAMPLE in file_names:
            file_path = os.path.join(BASE_DATA_PATH, file_names[self.SAMPLE])
            with mrcfile.open(file_path, permissive=True) as mrc:
                data = mrc.data

            proj = data.sum(0)
            target_size = self.N
            pad_h = target_size - proj.shape[0]
            pad_w = target_size - proj.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_proj = np.pad(proj, ((pad_top, pad_bottom),
                                (pad_left, pad_right)),
                                mode='constant', constant_values=0)
            mask = padded_proj
        return mask, do_fft(mask)

    def phase_ramp(self, shiftx, shifty):
        return np.exp(1j * 2.0 * np.pi * (self.qh * shiftx + self.qk * shifty))

    def unit_cell(self):
        ind = np.arange(self.N) - self.N//2
        rad = np.sqrt(ind[:,None]**2 + ind[None,:]**2)

        np.random.seed(self.SEED)
        random_ = np.random.rand(self.N, self.N)
        random_ = random_ > 0.7
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap')
        return unitc, self.SCALE * do_fft(unitc)

    def generate_dataset(self):
        tobj, ftobj = self.target_obj()
        unitc, funitc = self.unit_cell()
        ftobj = 10**(-4) * ftobj
        funitc = 10**(-4) * funitc
        intens_vals = np.zeros((self.NUM_FRAMES, self.N, self.N))
        rotated_ftobjs = np.zeros((self.NUM_FRAMES, self.N, self.N), dtype=np.complex128)

        for i in range(self.NUM_FRAMES):
            shiftx = self.dx_vals[i]
            shifty = self.dy_vals[i]
            fluence = self.fluence_vals[i]
            phase = self.phase_ramp(shiftx, shifty)

            ftobj_rot = rotate_arr(self.N, ftobj, self.angles[i])
            ft = fluence * ftobj_rot * phase
            rotated_ftobjs[i] = ftobj_rot

            intens = np.abs(funitc + ft)**2 + self.BACKGROUND**2

            if self.ADD_NOISE:
                intens_vals[i] = np.random.poisson(intens * self.EXPOSURE_TIME) / self.EXPOSURE_TIME - self.BACKGROUND**2
            else:
                intens_vals[i] = intens - self.BACKGROUND**2

        return intens_vals, rotated_ftobjs, ftobj, funitc, tobj, unitc

    def save_data(self, filename):
        intens_vals, rotated_ftobjs, ftobj, funitc, tobj, unitc = self.generate_dataset()
        print(f"Saving dataset to {filename}")
        try:
            with h5py.File(filename, 'w') as f:
                f.create_dataset('SCALE', data=self.SCALE)
                f.create_dataset('intens', data=intens_vals)
                f.create_dataset('rotated_ftobjs', data=rotated_ftobjs)
                f.create_dataset('ftobj', data=ftobj)
                f.create_dataset('tobj', data=tobj)
                f.create_dataset('funitc', data=funitc)
                f.create_dataset('unitc', data=unitc)
                f.create_dataset('shifts', data=np.vstack((self.dx_vals, self.dy_vals)).T)
                f.create_dataset('fluence', data=self.fluence_vals)
                f.create_dataset('angles', data=self.angles)
                f.create_dataset('exposure_time', data=self.EXPOSURE_TIME)
                f.create_dataset('background', data=self.BACKGROUND)
            print(f"Dataset saved to : {filename}")
        except Exception as e:
            print(f"Failed to save dataset to {filename}: {e}")

if __name__ == "__main__":
    config_file = 'config.ini'
    (N, SCALE, NUM_FRAMES, SEED, SAMPLE, ADD_NOISE, DATA_FILE, EXPOSURE_TIME, BACKGROUND_SQR) = makedata_config(config_file)
    generator = DataGenerator(N, SCALE, NUM_FRAMES, SEED, SAMPLE, ADD_NOISE, DATA_FILE, EXPOSURE_TIME, BACKGROUND_SQR)
    generator.save_data(DATA_FILE)
