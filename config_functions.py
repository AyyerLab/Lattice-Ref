import ast
import configparser

def makedata_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    N = config.getint('PARAMETERS', 'n')
    SCALE = config.getfloat('PARAMETERS', 'scale')
    NUM_FRAMES = config.getint('PARAMETERS', 'num_frames')
    SEED = config.getint('PARAMETERS', 'seed')
    SAMPLE = config.get('PARAMETERS', 'sample')
    
    ADD_NOISE = config.getboolean('NOISE', 'add')
    NOISE_K = config.getfloat('NOISE', 'k')

    DATA_FILE = config.get('FILES', 'DATA_FILE')
    return N, SCALE, NUM_FRAMES, SEED, SAMPLE, ADD_NOISE, NOISE_K, DATA_FILE

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
    USE_SHRINKWRAP = config.getboolean('SHRINKWRAP', 'use')
    
    DATA_FILE = config.get('FILES', 'data_file')
    OUTPUT_FILE = config.get('FILES', 'output_file')
    return N, NUM_FRAMES, NUM_ITER, SEED, SAMPLE, INIT_OBJ, PIXELS, USE_SHRINKWRAP, DATA_FILE, OUTPUT_FILE

