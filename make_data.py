import numpy as np
from scipy import ndimage
import h5py

import mrcfile
import configparser
import ast

from utils import makedata_config, do_fft, do_ifft

class DataGenerator:
    def __init__(self, N, num_samples, shape, scale, shifts, seed, add_noise, noise_mu, noise_k):
        self.N = N
        self.NUM_SAMPLES = num_samples
        self.SHAPE = shape
        self.SCALE = scale
        self.SHIFTS = shifts
        self.SEED = seed

        self.ADD_NOISE = add_noise
        self.NOISE_MU = noise_mu
        self.NOISE_K = noise_k

        self.cen = self.N // 2
        self.qh, self.qk = np.indices((self.N, self.N))
        self.qh -= self.cen
        self.qk -= self.cen

        self.xvals = np.linspace(0, 1, 500)
        self.max_fl = self.pfluence(self.xvals[1:]).max()
        np.random.seed(self.SEED)
        self.fluence_vals = self.randfluence(self.NUM_SAMPLES)
        print(self.fluence_vals)
        self.dx_vals = np.random.uniform(self.SHIFTS[0], self.SHIFTS[1], size=self.NUM_SAMPLES)
        self.dy_vals = np.random.uniform(self.SHIFTS[0], self.SHIFTS[1], size=self.NUM_SAMPLES)



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

        elif self.SHAPE == 'PS1':
            with mrcfile.open('/scratch/mallabhi/lattice_ref/data/PS1_map.mrc', permissive=True) as mrc:
                ps1 = mrc.data

            ps1_proj= ps1.sum(0)
            pad_h = self.N - ps1_proj.shape[0]
            pad_w = self.N - ps1_proj.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_ps1_proj = np.pad(ps1_proj, ((pad_top, pad_bottom), 
                                                (pad_left, pad_right)), 
                                                mode='constant', constant_values=0)
            mask = padded_ps1_proj
    
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
        noise_vals = None
        if self.ADD_NOISE:
            noise_vals = np.zeros_like(intens_vals)

        for i in range(self.NUM_SAMPLES):
            shiftx = self.dx_vals[i]
            shifty = self.dy_vals[i]
            fluence = self.fluence_vals[i]
            phase = self.phase_ramp(shiftx, shifty)
            intens = np.abs(funitc + fluence * ftobj * phase)**2

            if self.ADD_NOISE:
                noise_sigma = self.NOISE_K * np.sqrt(intens)
                np.random.seed(self.SEED)
                noise = np.random.normal(loc=self.NOISE_MU, scale=noise_sigma, size=intens.shape)
                noise_vals[i] = noise
                intens_vals[i] = intens + noise
            else:
                intens_vals[i] = intens

        return intens_vals, noise_vals, ftobj, funitc, tobj, unitc

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
            f.create_dataset('SCALE', data=self.SCALE)

            if self.ADD_NOISE:
                f.create_dataset('noise', data=noise)
            else:
                f.create_dataset('noise', data=np.empty((0,)))

if __name__ == "__main__":
    config_file = 'config.ini'
    (N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, 
     SEED, DATA_FILE, ADD_NOISE, NOISE_MU, NOISE_K) = makedata_config(config_file)
    generator = DataGenerator(N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, 
                              SEED, ADD_NOISE, NOISE_MU, NOISE_K)
    generator.save_data(DATA_FILE)

