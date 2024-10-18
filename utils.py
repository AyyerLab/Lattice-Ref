import configparser
import ast
import numpy as np

def get_vals(array, cen, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]

def do_fft(obj):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

def init_tobj(N, pixels):
    size = (N, N)
    obj = np.zeros(size)
    cen = (N // 2, N // 2)
    y, x = np.ogrid[:N, :N]
    dcen = np.sqrt((x - cen[1])**2 + (y - cen[0])**2)
    obj.ravel()[np.argsort(dcen.ravel())[:pixels]] = 1
    return obj

def makedata_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    N = config.getint('DATA_GENERATION', 'N')
    NUM_SAMPLES = config.getint('DATA_GENERATION', 'NUM_SAMPLES')
    SCALE = config.getint('DATA_GENERATION', 'SCALE')
    SEED = config.getint('DATA_GENERATION', 'SEED')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')
    SHAPE = config.get('DATA_GENERATION', 'SHAPE')
    SHIFTS = ast.literal_eval(config['DATA_GENERATION']['SHIFTS'])
    FLUENCE = ast.literal_eval(config['DATA_GENERATION']['FLUENCE'])
    ANGLES = ast.literal_eval(config['DATA_GENERATION']['ANGLES'])

    APPLY_ORIENTATION = config.getboolean('DATA_GENERATION', 'APPLY_ORIENTATION')

    ADD_NOISE = config.getboolean('DATA_GENERATION', 'ADD_NOISE')
    NOISE_MU = config.getfloat('DATA_GENERATION', 'NOISE_MU')
    NOISE_K = config.getfloat('DATA_GENERATION', 'NOISE_K')
    return N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, SEED, DATA_FILE, ADD_NOISE, NOISE_MU, NOISE_K, ANGLES, APPLY_ORIENTATION

def run_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    N = config.getint('DATA_GENERATION', 'N')
    NUM_ITER = config.getint('OPTIMIZATION', 'NUM_ITERATION')
    INIT_FTOBJ = config.get('OPTIMIZATION', 'INIT_FTOBJ')

    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')
    OUTPUT_FILE = config.get('OPTIMIZATION', 'OUTPUT_FILE')
    PIXELS = config.getint('OPTIMIZATION', 'PIXELS')
    USE_SHRINKWRAP = config.getboolean('OPTIMIZATION', 'USE_SHRINKWRAP')
    SHRINKWRAP_RUNS = ast.literal_eval(config['OPTIMIZATION']['SHRINKWRAP_RUNS'])

    ANGLES = ast.literal_eval(config['DATA_GENERATION']['ANGLES'])
    NUM_SAMPLES = config.getint('DATA_GENERATION', 'NUM_SAMPLES')
    return N, NUM_ITER, INIT_FTOBJ, DATA_FILE, OUTPUT_FILE, PIXELS, USE_SHRINKWRAP, SHRINKWRAP_RUNS, ANGLES, NUM_SAMPLES

