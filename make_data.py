import numpy as np
from scipy import ndimage
import h5py

import configparser
import ast

from utils import makedata_config, do_fft, do_ifft

class DataGenerator:
    def __init__(self, N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, SEED, ADD_NOISE, NOISE_MU, NOISE_K):
        self.N = N
        self.NUM_SAMPLES = NUM_SAMPLES
        self.SHAPE = SHAPE
        self.SCALE = SCALE
        self.SHIFTS = SHIFTS
        self.FLUENCE = FLUENCE
        self.SEED = SEED

        self.ADD_NOISE = ADD_NOISE
        self.NOISE_MU = NOISE_MU
        self.NOISE_K = NOISE_K

        self.cen = N // 2
        self.qh, self.qk = np.indices((N, N))
        self.qh -= self.cen
        self.qk -= self.cen

        np.random.seed(SEED)
        self.dx_vals = np.random.uniform(SHIFTS[0], SHIFTS[1], size=NUM_SAMPLES)
        self.dy_vals = np.random.uniform(SHIFTS[0], SHIFTS[1], size=NUM_SAMPLES)
        self.fluence_vals = np.random.uniform(FLUENCE[0], FLUENCE[1], size=NUM_SAMPLES)

    def target_obj(self):
        size = self.N
        mcen = size // 2
        x, y = np.indices((size, size), dtype='f8')
        mask = np.zeros((size, size), dtype='f8')

        if self.SHAPE == 'cluster':
            num_circ = 55
            for i in range(num_circ):
                rad = (0.7 + 0.3 * (np.cos(2.5 * i) - np.sin(i / 4.0))) * size / 20.0
                cen = [
                    (3 * np.sin(2 * i) + 0.5 * np.sin(i / 2.0)) * size / 45.0 + mcen * 4.0 / 4.4,
                    (0.5 * np.cos(i / 2.0) + 3 * np.cos(i / 2.0)) * size / 45.0 + mcen * 4.0 / 4.4
                ]
                diskrad = np.sqrt((x - cen[0])**2 + (y - cen[1])**2)
                inside = diskrad <= rad
                mask[inside] += 1.0 - (diskrad[inside] / rad)**2
        elif self.SHAPE == 'rect':
            mask[45:70, 40:70] = 1

        return mask, do_fft(mask)

    def phase_ramp(self, shiftx, shifty):
        return np.exp(1j * 2.0 * np.pi * (self.qh * shiftx + self.qk * shifty))

    def unit_cell(self):
        np.random.seed(SEED)
        random_ = np.random.rand(self.N, self.N)
        random_ = random_ > 0.7
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap')

        return unitc, self.SCALE * do_fft(unitc)

    def generate_dataset(self):
        # Target Object
        tobj, ftobj = self.target_obj()

        # Unit Cell
        unitc, funitc = self.unit_cell()

        # Generate Data
        intens_vals = np.zeros((self.NUM_SAMPLES, self.N, self.N))
        for i in range(self.NUM_SAMPLES):
            shiftx = self.dx_vals[i]
            shifty = self.dy_vals[i]
            fluence = self.fluence_vals[i]
            phase = self.phase_ramp(shiftx, shifty)
            intens = np.abs(funitc + fluence * ftobj * phase)**2
            
            if self.ADD_NOISE:
                noise_sigma = self.NOISE_K * np.sqrt(intens)
                np.random.seed(SEED)
                noise = np.random.normal(loc=self.NOISE_MU, scale=noise_sigma, size=intens.shape)
                intens_n = intens + noise
            intens_vals[i] = intens_n

        return intens_vals, noise, ftobj, funitc, tobj, unitc

    def save_data(self, filename):
        intens_vals, noise, ftobj, funitc, tobj, unitc = self.generate_dataset()
        with h5py.File(filename, 'w') as f:
            f.create_dataset('intens', data=intens_vals)
            f.create_dataset('ftobj', data=ftobj)
            f.create_dataset('tobj', data=tobj)
            f.create_dataset('funitc', data=funitc)
            f.create_dataset('unitc', data=unitc)
            f.create_dataset('shifts', data=np.vstack((self.dx_vals, self.dy_vals)).T)
            f.create_dataset('fluence', data=self.fluence_vals)
            f.create_dataset('NOISE_K', data=self.NOISE_K)
            f.create_dataset('noise', data=noise)

if __name__ == "__main__":
    config_file = 'config.ini'
    (N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, 
     SEED, DATA_FILE, ADD_NOISE, NOISE_MU, NOISE_K) = makedata_config(config_file)
    generator = DataGenerator(N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, 
                              SEED, ADD_NOISE, NOISE_MU, NOISE_K)
    generator.save_data(DATA_FILE)

