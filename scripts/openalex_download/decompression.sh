#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=100G       
#SBATCH --time=20:00:00   
#SBATCH --job-name decompression
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

cd PaperSeek
~/.pyenv1/bin/pyenv local 3.12.8
poetry env use /work/msakni2s/.pyenv/versions/3.12.8/bin/python3.12
poetry config keyring.enabled false
poetry install --no-root --without dev
poetry run python scripts/openalex_download/decompression.py
