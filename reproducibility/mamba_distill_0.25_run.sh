#!/bin/bash
#SBATCH -J mamba_distill_0.25
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container_mamba.toml
#SBATCH --account=a-a10
#SBATCH -e mamba_distill_0.25_logs/run.error
#SBATCH -o mamba_distill_0.25_logs/run.out
pwd
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba/train_hybrid.py mamba_llama/llama3_0.25_mamba.yaml
