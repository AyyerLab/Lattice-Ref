import numpy as np
from scipy import ndimage
import h5py
import configparser

class DataGenerator:
    def __init__(self, N, NUM_SAMPLES, SHAPE='rect', SCALE=4, SEED=42):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.SHAPE = SHAPE
        self.SCALE = SCALE
        self.SEED = SEED

        self.N = N
        self.cen = N // 2
        self.qh, self.qk = np.indices((N, N))
        self.qk -= self.cen
        self.qh -= self.cen

        np.random.seed(SEED)
        self.dx_vals = np.random.uniform(0, 1, size=NUM_SAMPLES)
        self.dy_vals = np.random.uniform(0, 1, size=NUM_SAMPLES)
        self.fluence_vals = np.random.uniform(0.1, 1, size=NUM_SAMPLES)
        #self.fluence_vals = np.random.uniform(0.5, 1.5, size=NUM_SAMPLES)

    def target_obj(self):
        tobj = np.zeros((self.N, self.N))
        if self.SHAPE == 'X':
            tobj[25:80, self.N//2-5:self.N//2+5] = 1
            tobj[20:40, self.N//2-20:self.N//2+20] = 1
            tobj[50:60, self.N//2-20:self.N//2+20] = 1
        elif self.SHAPE == 'rect':
            tobj[45:70, 40:70] = 1
        return tobj, np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tobj)))

    def phase_ramp(self, shiftx, shifty):
        return np.exp(1j * 2.0 * np.pi * (self.qh * shiftx + self.qk * shifty))

    def translate(self, ftobj, shiftx, shifty):
        pramp = ftobj * self.phase_ramp(shiftx, shifty)
        return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ftobj * pramp)))

    def unit_cell(self):
        np.random.seed(self.SEED)
        random_ = np.random.rand(self.N, self.N)
        random_ = random_ > 0.7
        unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(1.75, 1.25), mode='wrap')
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
        return intens_vals, ftobj, funitc

    def save_data(self, filename):
        intens_vals, ftobj, funitc = self.generate_dataset()
        with h5py.File(filename, 'w') as f:
            f['intens'] = intens_vals
            f['ftobj'] = ftobj
            f['funitc'] = funitc
            f['shifts'] = np.vstack((self.dx_vals, self.dy_vals)).T
            f['fluence'] = self.fluence_vals

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    N = config.getint('DATA_GENERATION', 'N')
    NUM_SAMPLES = config.getint('DATA_GENERATION', 'NUM_SAMPLES')
    SHAPE = config.get('DATA_GENERATION', 'SHAPE')
    SCALE = config.getint('DATA_GENERATION', 'SCALE')
    SEED = config.getint('DATA_GENERATION', 'SEED')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')
    return N, NUM_SAMPLES, SHAPE, SCALE, SEED, DATA_FILE

if __name__ == "__main__":
    config_file = 'config.ini'
    N, NUM_SAMPLES, SHAPE, SCALE, SEED, DATA_FILE = load_config(config_file)
    generator = DataGenerator(N, NUM_SAMPLES, SHAPE, SCALE, SEED)
    generator.save_data(DATA_FILE)

