import configparser
import ast


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

    return N, NUM_SAMPLES, SHAPE, SCALE, SHIFTS, FLUENCE, SEED, DATA_FILE

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

    return N, NUM_SAMPLES, SCALE, SEED, INIT_FTOBJ, DATA_FILE, OUTPUT_FILE

def get_vals(array, cen, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]

