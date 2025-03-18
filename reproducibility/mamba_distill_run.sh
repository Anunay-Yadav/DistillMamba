#!/bin/bash
#SBATCH -J mamba_distill
#SBATCH -t 24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container.toml
#SBATCH --account=a-a10
#SBATCH -e mamba_distill_node8_logs/run.error
#SBATCH -o mamba_distill_node8_logs/run.out
######################
#### Set network #####
######################
export GPUS_PER_NODE=4
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --config_file multi_gpu.yaml
    "
export CMD="$LAUNCHER train_mamba/train_compressed.py mamba_distill_yaml/mamba_noinit.yaml"
srun $CMD