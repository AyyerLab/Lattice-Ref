import configparser
import subprocess
import os

B_sqr_vals = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
exposure_times = [1e7, 1e8]
seed_values = [52, 62, 72, 82, 92, 102, 112, 122]

base_config_file = 'config.ini'
data_dir = '/scratch/mallabhi/lattice_ref/data/K'
os.makedirs(data_dir, exist_ok=True)

run_number = 102

for b in B_sqr_vals:
    for et in exposure_times:
        for seed in seed_values:
            config = configparser.ConfigParser()
            config.read(base_config_file)

            config['PARAMETERS']['seed'] = str(seed)
            config['PARAMETERS']['exposure_time'] = str(et)
            config['PARAMETERS']['background_sqr'] = str(b)

            dataset_filename = f'dataset_run{run_number:03d}.h5'
            data_file_path = os.path.join(data_dir, dataset_filename)
            config['FILES']['data_file'] = data_file_path

            with open(base_config_file, 'w') as configfile:
                config.write(configfile)

            print(f"Generating dataset for B^2: {b}, exposure_time: {et}, seed: {seed}, run number: {run_number}")

            try:
                subprocess.run(['python', 'make_data.py', base_config_file], check=True)
                if os.path.exists(data_file_path):
                    print(f"Dataset successfully created: {data_file_path}")
                else:
                    print(f"Data generation script ran but file was not found: {data_file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Data generation failed for run number {run_number}: {e}")

            run_number += 1

