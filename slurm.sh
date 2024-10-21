#!/bin/sh
#SBATCH --job-name            optim
#SBATCH --output              log/slurm/%j.out
#SBATCH --cpus-per-task       1
#SBATCH --gpus                1
#SBATCH --mem                 85G
#SBATCH --ntasks              1
#SBATCH --partition           gpu
#SBATCH --time                24:00:00


source load_module.sh
module load cnicuda
module load gcc/11
module load openmpi/4
module load mpi4py
python run_optimization.py
