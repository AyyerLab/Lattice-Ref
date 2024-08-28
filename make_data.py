import numpy as np
from scipy import ndimage
import h5py

import configparser
import ast

from utils import makedata_config

class DataGenerator:
    def __init__(self, N, NUM_SAMPLES, SHAPE='rect', SCALE=4, SHIFTS=[0,1], FLUENCE=[0,1], SEED=42):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.SHAPE = SHAPE
        self.SCALE = SCALE
        self.SHIFTS = SHIFTS
        self.FLUENCE  = FLUENCE
        self.SEED = SEED

        self.N = N
        self.cen = N // 2
        self.qh, self.qk = np.indices((N, N))
        self.qk -= self.cen
        self.qh -= self.cen

        np.random.seed(SEED)
        self.dx_vals = np.random.uniform(SHIFTS[0], SHIFTS[1], size=NUM_SAMPLES)
        self.dy_vals = np.random.uniform(SHIFTS[0], SHIFTS[1], size=NUM_SAMPLES)
        self.fluence_vals = np.random.uniform(FLUENCE[0], FLUENCE[1], size=NUM_SAMPLES)


    def target_obj(self, HETRO=False, CREATE_RAND=False):
        size = self.N
        mcen = size // 2
        x, y = np.indices((size, size), dtype='f8')
        mask = np.zeros((size, size), dtype='f8')

        if self.SHAPE == 'cluster':
            num_circ = 55 if not HETRO else 25
            for i in range(num_circ):
                if CREATE_RAND:
                    rad = (0.7 + 0.3 * np.random.rand()) * size / 25.0
                    cen = np.random.rand(2) * size / 5.0 + mcen * 4.0 / 5.0
                else:
                    rad = (0.7 + 0.3 * (np.cos(2.5 * i) - np.sin(i / 4.0))) * size / 20.0
                    cen = [
                        (3 * np.sin(2 * i) + 0.5 * np.sin(i / 2.0)) * size / 45.0 + mcen * 4.0 / 4.4,
                        (0.5 * np.cos(i / 2.0) + 3 * np.cos(i / 2.0)) * size / 45.0 + mcen * 4.0 / 4.4
                        ]
                diskrad = np.sqrt((x - cen[0])**2 + (y - cen[1])**2)
                mask[diskrad <= rad] += 1.0 - (diskrad[diskrad <= rad] / rad)**2
        elif self.SHAPE == 'rect':
                mask[45:70, 40:70] = 1
        return mask, np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(mask)))


    def phase_ramp(self, shiftx, shifty):
        return np.exp(1j * 2.0 * np.pi * (self.qh * shiftx + self.qk * shifty))

    def translate(self, ftobj, shiftx, shifty):
        pramp = ftobj * self.phase_ramp(shiftx, shifty)
        return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ftobj * pramp)))

    def unit_cell(self):
        np.random.seed(self.SEED)
        random_ = np.random.rand(self.N, self.N)
        random_ = random_ > 0.7
        #unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(1.75, 1.25), mode='wrap')
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap')
        return unitc, np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(unitc)))

    def generate_dataset(self):
        # Target Object
        tobj, ftobj = self.target_obj()

        # Unit Cell
        unitc, funitc = self.unit_cell()

        # Generate Data
        intens_vals = np.zeros((self.NUM_SAMPLES, self.N, self.N))
        for i in range(self.NUM_SAMPLES):
            intens = self.fluence_vals[i] * np.abs(self.SCALE * funitc + ftobj * self.phase_ramp(self.dx_vals[i], self.dy_vals[i]))**2
            intens_vals[i] = intens
        return intens_vals, ftobj, funitc, tobj, unitc

    def save_data(self, filename):
        intens_vals, ftobj, funitc, tobj, unitc = self.generate_dataset()
        with h5py.File(filename, 'w') as f:
            f['intens'] = intens_vals
            f['ftobj'] = ftobj
            f['tobj'] = tobj
            f['funitc'] = funitc
            f['unitc'] = unitc
            f['shifts'] = np.vstack((self.dx_vals, self.dy_vals)).T
            f['fluence'] = self.fluence_vals


if __name__ == "__main__":
    config_file = 'config.ini'
    N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, SEED, DATA_FILE = makedata_config(config_file)
    generator = DataGenerator(N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, SEED)
    generator.save_data(DATA_FILE)

