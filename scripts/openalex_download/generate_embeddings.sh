#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=100G       
#SBATCH --time=24:00:00   
#SBATCH --job-name embeddings
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
module load cuda
ARG=${1:-0}

cd PaperSeek
~/.pyenv1/bin/pyenv local 3.12.8
poetry config keyring.enabled false
poetry install --no-root --without dev
poetry run python scripts/openalex_download/generate_embeddings.py -f ${ARG}

#sbatch generate_embeddings.sh
# sinfo --partition=wr44 --format="%10N %.6D %10P %.4c %.8z %8O %.8m %10e %12C" --iterate=5
# sinfo --partition=hpc1 --nodes=wr64 --format="%10N %.6D %10P %.4c %.8z %8O %.8m %10e %14C" --iterate=5  
