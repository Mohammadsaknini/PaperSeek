#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=70G       
#SBATCH --nodelist=wr25
#SBATCH --time=1-00:00:00   
#SBATCH --job-name stella_embeddings_6
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
module load cuda

cd PaperSeek
~/.pyenv1/bin/pyenv local 3.12.8
poetry env use /home/msakni2s/.pyenv/versions/3.12.8/bin/python3.12
poetry config keyring.enabled false
poetry install --no-root --without dev
poetry run python scripts/benchmark_embedding_models/script.py -g 0 -i 6

#sbatch generate_embeddings.sh
# sinfo --partition=wr44 --format="%10N %.6D %10P %.4c %.8z %8O %.8m %10e %12C" --iterate=5
# sinfo --partition=hpc1 --nodes=wr64 --format="%10N %.6D %10P %.4c %.8z %8O %.8m %10e %14C" --iterate=5  
