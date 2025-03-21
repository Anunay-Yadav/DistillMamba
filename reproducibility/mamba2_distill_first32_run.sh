#!/bin/bash
#SBATCH -J mamba2_distill_first32
#SBATCH -t 24:00:00
#SBATCH --nodes=8
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/my_repo/DistillMamba/starter_container_mamba2.toml
#SBATCH --account=a-a10
#SBATCH -e mamba2_distill_first32_logs/run.error
#SBATCH -o mamba2_distill_first32_logs/run.out
######################
#### Set network #####
######################
export GPUS_PER_NODE=4
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
export ACCELERATE_LOG_LEVEL=info
export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --config_file multi_gpu.yaml
    "
export CMD="$LAUNCHER train_mamba2/train_compressed.py mamba2_distill_yaml/mamba2_init_first32.yaml"
cd lm-evaluation-harness/
pip install -e .
cd ..
pip install datasets==2.20.0
srun $CMD