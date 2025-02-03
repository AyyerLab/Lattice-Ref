import numpy as np
from scipy import ndimage
import h5py

import mrcfile
import configparser
import ast

from config_functions import makedata_config
from utils import do_fft, do_ifft
from utils import rotate_arr

class DataGenerator:
    def __init__(self, n, scale, num_frames, seed, shifts, sample, fluence_mode, apply_orientation, add_noise, noise_mu, noise_k, data_file):
        self.N = N
        self.NUM_FRAMES = num_frames
        self.SAMPLE = sample
        self.SCALE = scale
        self.SEED = seed
        self.SHIFTS = shifts
        self.FLUENCE_MODE = fluence_mode

        self.ADD_NOISE = add_noise
        self.NOISE_MU = noise_mu
        self.NOISE_K = noise_k

        self.APPLY_ORIENTATION = apply_orientation

        self.DATA_FILE = data_file

        self.cen = self.N // 2
        self.qh, self.qk = np.indices((self.N, self.N))
        self.qh -= self.cen
        self.qk -= self.cen

        if self.FLUENCE_MODE == "distribution":
            xvals = np.linspace(0, 1, 500)
            self.max_fl = self.pfluence(xvals[1:]).max()
            np.random.seed(self.SEED)
            self.fluence_vals = self.randfluence(self.NUM_FRAMES)
        else:
            raise ValueError(f"Invalid FLUENCE mode: {FLUENCE_MODE}")

        np.random.seed(self.SEED)
        self.dx_vals = np.random.uniform(self.SHIFTS[0], self.SHIFTS[1], size=self.NUM_FRAMES)
        self.dy_vals = np.random.uniform(self.SHIFTS[0], self.SHIFTS[1], size=self.NUM_FRAMES)

        if self.APPLY_ORIENTATION:
            np.random.seed(self.SEED)
            self.angles = np.random.uniform(0, 1, size=self.NUM_FRAMES) * 2. * np.pi
            #self.angles = np.random.uniform(0, 1, size=self.NUM_FRAMES) * (2. * np.pi / 3)

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

        if self.SAMPLE in ['PS1_5Angs','PS1_10Angs','RIB_5Angs','RIB_10Angs']:
            file_paths = {'PS1_5Angs': '/scratch/mallabhi/lattice_ref/data/PS1_map.mrc',
                          'PS1_10Angs': '/scratch/mallabhi/lattice_ref/data/PS1_10angs.mrc',
                          'RIB_5Angs': '/scratch/mallabhi/lattice_ref/data/rib.mrc',
                          'RIB_10Angs': '/scratch/mallabhi/lattice_ref/data/Rib_10angs.mrc'}
            target_file = file_paths[self.SAMPLE]
            with mrcfile.open(target_file, permissive=True) as mrc:
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
        #unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap') * (rad<30)
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap')
        return unitc, self.SCALE * do_fft(unitc)

    def generate_dataset(self):
        # Target Object
        tobj, ftobj = self.target_obj()
        # Unit Cell
        unitc, funitc = self.unit_cell()
        # Generate Data
        intens_vals = np.zeros((self.NUM_FRAMES, self.N, self.N))
        noise_vals = None
        rotated_ftobjs = np.zeros((self.NUM_FRAMES, self.N, self.N), dtype=np.complex128)

        if self.ADD_NOISE:
            noise_vals = np.zeros_like(intens_vals)

        for i in range(self.NUM_FRAMES):
            shiftx = self.dx_vals[i]
            shifty = self.dy_vals[i]
            fluence = self.fluence_vals[i]
            phase = self.phase_ramp(shiftx, shifty)

            if self.APPLY_ORIENTATION:
                ftobj_rot = rotate_arr(self.N, ftobj, self.angles[i])
                ft = fluence * ftobj_rot * phase
                rotated_ftobjs[i] = ftobj_rot
            else:
                ft = fluence * ftobj * phase

            intens = np.abs(funitc + ft)**2

            if self.ADD_NOISE:
                noise_sigma = self.NOISE_K * np.sqrt(intens)
                np.random.seed(self.SEED)
                noise = np.random.normal(loc=self.NOISE_MU, scale=noise_sigma, size=intens.SAMPLE)
                noise_vals[i] = noise
                intens_vals[i] = intens + noise
            else:
                intens_vals[i] = intens

        return intens_vals, noise_vals, rotated_ftobjs, ftobj, funitc, tobj, unitc


    def save_data(self, filename):
        intens_vals, noise, rotated_ftobjs, ftobj, funitc, tobj, unitc = self.generate_dataset()
        print(f"Saving dataset to {filename}")
        try:
            with h5py.File(filename, 'w') as f:
                f.create_dataset('intens', data=intens_vals)
                f.create_dataset('ftobj', data=ftobj)
                f.create_dataset('rotated_ftobjs', data=rotated_ftobjs)
                f.create_dataset('tobj', data=tobj)
                f.create_dataset('funitc', data=funitc)
                f.create_dataset('unitc', data=unitc)
                f.create_dataset('shifts', data=np.vstack((self.dx_vals, self.dy_vals)).T)
                f.create_dataset('fluence', data=self.fluence_vals)
                f.create_dataset('NOISE_K', data=self.NOISE_K)
                f.create_dataset('SCALE', data=self.SCALE)

                if self.APPLY_ORIENTATION:
                    f.create_dataset('angles', data=self.angles)

                if self.ADD_NOISE:
                    f.create_dataset('noise', data=noise)
                else:
                    f.create_dataset('noise', data=np.empty((0,)))
            print(f"Dataset saved to : {filename}")
        except Exception as e:
            print(f"Failed to save dataset to {filename}: {e}")

if __name__ == "__main__":
    config_file = 'config.ini'
    ( N, SCALE, NUM_FRAMES, SEED, SHIFTS, SAMPLE, FLUENCE_MODE, APPLY_ORIENTATION, ADD_NOISE, NOISE_MU, NOISE_K, DATA_FILE) = makedata_config(config_file)

    generator = DataGenerator( N, SCALE, NUM_FRAMES, SEED, SHIFTS, SAMPLE, FLUENCE_MODE, APPLY_ORIENTATION, ADD_NOISE, NOISE_MU, NOISE_K, DATA_FILE)

    generator.save_data(DATA_FILE)

