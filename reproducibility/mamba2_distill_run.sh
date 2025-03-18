#!/bin/bash
#SBATCH -J mamba2_distill
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container.toml
#SBATCH --account=a-a10
#SBATCH -e mamba2_distill_logs/run.error
#SBATCH -o mamba2_distill_logs/run.out
pwd
ulimit -c 0
pip install transformers -U
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba2/train_compressed.py mamba2_distill_yaml/mamba2_noinit.yaml
