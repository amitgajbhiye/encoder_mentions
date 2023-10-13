#!/bin/bash --login

#SBATCH --job-name=setup3_mention_embeds_mcrae_concepts

#SBATCH --output=logs/get_mention_embeddings/out_setup3_mention_embeds_mcrae_concepts.txt
#SBATCH --error=logs/get_mention_embeddings/err_setup3_mention_embeds_mcrae_concepts.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/setup3_mention_embeds_mcrae_concepts.json

echo 'Job Finished !!!'