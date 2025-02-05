import ast
import configparser

def makedata_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    N = config.getint('PARAMETERS', 'n')
    SCALE = config.getfloat('PARAMETERS', 'scale')
    NUM_FRAMES = config.getint('PARAMETERS', 'num_frames')
    SEED = config.getint('PARAMETERS', 'seed')
    SHIFTS = ast.literal_eval(config['PARAMETERS']['shifts'])

    SAMPLE = config.get('OBJECT', 'sample')

    FLUENCE_MODE = config.get('FLUENCE', 'mode')
    
    APPLY_ORIENTATION = config.getboolean('ORIENTATION', 'apply')
    
    ADD_NOISE = config.getboolean('NOISE', 'add')
    NOISE_MU = config.getfloat('NOISE', 'mu')
    NOISE_K = config.getfloat('NOISE', 'k')

    DATA_FILE = config.get('FILES', 'DATA_FILE')
    return N, SCALE, NUM_FRAMES, SEED, SHIFTS, SAMPLE, FLUENCE_MODE, APPLY_ORIENTATION, ADD_NOISE, NOISE_MU, NOISE_K, DATA_FILE

def run_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    N = config.getint('PARAMETERS', 'n')
    NUM_FRAMES = config.getint('PARAMETERS', 'num_frames')
    SEED = config.getint('PARAMETERS', 'seed')
    
    NUM_ITER = config.getint('OPTIMIZATION', 'num_iteration')
    INIT_OBJ = config.get('OPTIMIZATION', 'init_obj')
    APPLY_ORIENTATION = config.getboolean('ORIENTATION', 'apply')
    PIXELS = config.getint('OPTIMIZATION', 'pixels')
    USE_SHRINKWRAP = config.getboolean('SHRINKWRAP', 'use')
    
    DATA_FILE = config.get('FILES', 'data_file')
    OUTPUT_FILE = config.get('FILES', 'output_file')
    return N, NUM_FRAMES, NUM_ITER, SEED, INIT_OBJ, PIXELS, USE_SHRINKWRAP, APPLY_ORIENTATION,DATA_FILE, OUTPUT_FILE

