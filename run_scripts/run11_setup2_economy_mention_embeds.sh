#!/bin/bash --login

#SBATCH --job-name=s2economy_mention_embeddings

#SBATCH --output=logs/get_mention_embeddings/out_setup2_economy_mention_embeddings.txt
#SBATCH --error=logs/get_mention_embeddings/err_setup2_economy_mention_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/small_onto_mention/setup2_economy_mention_embeddings.json

echo 'Job Finished !!!'