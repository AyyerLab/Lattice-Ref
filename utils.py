import configparser
import ast
import numpy as np

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

def optim_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    N = config.getint('DATA_GENERATION', 'N')
    NUM_SAMPLES = config.getint('DATA_GENERATION', 'NUM_SAMPLES')
    SCALE = config.getint('DATA_GENERATION', 'SCALE')
    SEED = config.getint('DATA_GENERATION', 'SEED')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')

    INIT_FTOBJ = config.get('OPTIMIZATION', 'INIT_FTOBJ')
    OUTPUT_FILE = config.get('OPTIMIZATION', 'OUTPUT_FILE')
    PIXELS = config.getint('OPTIMIZATION', 'PIXELS')

    return N, NUM_SAMPLES, SCALE, SEED, INIT_FTOBJ, DATA_FILE, OUTPUT_FILE, PIXELS

def get_vals(array, cen, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]


def do_fft(obj):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

