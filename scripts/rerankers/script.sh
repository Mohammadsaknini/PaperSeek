#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=100G       
#SBATCH --time=24:00:00   
#SBATCH --job-name rerankers
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
module load cuda
ARG=${1:-0}

cd PaperSeek
~/.pyenv1/bin/pyenv local 3.12.8
poetry config keyring.enabled false
poetry install --no-root --without dev
poetry run python scripts/rerankers/main.py -f ${ARG}