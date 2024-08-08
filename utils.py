import numpy as np
import h5py
import configparser

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    N = config.getint('DATA_GENERATION', 'N')
    SCALE = config.getint('DATA_GENERATION', 'SCALE')
    DATA_FILE = config.get('DATA_GENERATION', 'DATA_FILE')
    OUTPUT_FILE = config.get('OPTIMIZATION', 'OUTPUT_FILE', fallback=None)
    return N, DATA_FILE, SCALE, OUTPUT_FILE

def get_vals(array, cen, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]

