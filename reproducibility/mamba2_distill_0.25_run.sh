#!/bin/bash
#SBATCH -J mamba2_distill_0.25
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container.toml
#SBATCH --account=a-a10
#SBATCH -e mamba2_distill_0.25_logs/mamba_distill_run.error
#SBATCH -o mamba2_distill_0.25_logs/mamba_distill_run.out
pwd
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba2/train_hybrid.py mamba2_llama/llama3_0.25_mamba2.yaml
