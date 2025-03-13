#!/bin/bash
#SBATCH -J MambaInLamma_dockerfile_generator
#SBATCH -t 5:00:00
#SBATCH --nodes=1
#SBATCH --account=a-a10
podman build -t temp .
enroot import -x mount -o MambaInLamma-pytorch-23.10-py3-venv.sqsh podman://temp
