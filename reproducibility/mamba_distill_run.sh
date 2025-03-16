#!/bin/bash
#SBATCH -J mamba_distill
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container.toml
#SBATCH --account=a-a10
#SBATCH -e mamba_mamba_distill_temp_logs/mamba_distill_run.error
#SBATCH -o mamba_mamba_distill_temp_logs/mamba_distill_run.out
pwd
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba/train_mamba_compressed.py mamba_mamba/mamba_mamba.yaml
