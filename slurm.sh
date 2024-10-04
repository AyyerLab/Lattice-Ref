#!/bin/bash -l

#SBATCH --partition=p.ada
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=none
#SBATCH --time=24:00:00
#SBATCH -J optim
#SBATCH -o .%j.out
#SBATCH -e .%j.out

module purge
module load anaconda/3/2021.11 cuda/11.6
source activate /u/kayyer/conda-envs/cnicuda
module load gcc/11
module load openmpi/4
module load mpi4py
python run_optimization.py
