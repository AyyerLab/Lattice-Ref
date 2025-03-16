import ast
import configparser

def makedata_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    N = config.getint('PARAMETERS', 'N')
    SCALE = config.getfloat('PARAMETERS', 'SCALE')
    NUM_FRAMES = config.getint('PARAMETERS', 'NUM_FRAMES')
    SEED = config.getint('PARAMETERS', 'SEED')
    SAMPLE = config.get('PARAMETERS', 'SAMPLE')
    EXPOSURE_TIME = config.getfloat('PARAMETERS', 'EXPOSURE_TIME', fallback=1.0)
    BACKGROUND_SQR = config.getfloat('PARAMETERS', 'BACKGROUND_SQR', fallback=1.0)
    ADD_NOISE = config.getboolean('PARAMETERS', 'ADD_NOISE')
    DATA_FILE = config.get('FILES', 'DATA_FILE')
    return N, SCALE, NUM_FRAMES, SEED, SAMPLE, ADD_NOISE, DATA_FILE, EXPOSURE_TIME, BACKGROUND_SQR

def run_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    N = config.getint('PARAMETERS', 'n')
    NUM_FRAMES = config.getint('PARAMETERS', 'num_frames')
    SEED = config.getint('PARAMETERS', 'seed')
    SAMPLE = config.get('PARAMETERS', 'sample')

    NUM_ITER = config.getint('OPTIMIZATION', 'num_iteration')
    INIT_OBJ = config.get('OPTIMIZATION', 'init_obj')
    PIXELS = config.getint('OPTIMIZATION', 'pixels')

    DATA_FILE = config.get('FILES', 'data_file')
    OUTPUT_FILE = config.get('FILES', 'output_file')
    return N, NUM_FRAMES, NUM_ITER, SEED, SAMPLE, INIT_OBJ, PIXELS, DATA_FILE, OUTPUT_FILE

