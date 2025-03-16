#!/bin/bash
#SBATCH -J dataset_generator
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container.toml
#SBATCH --account=a-a10
pwd
pip install datasets

python dataset.py