#!/bin/bash --login

#SBATCH --job-name=mention_embeds_mcrae_concepts_with_mentions

#SBATCH --output=logs/get_mention_embeddings/out_mention_embeds_mcrae_concepts_with_mentions.txt
#SBATCH --error=logs/get_mention_embeddings/err_mention_embeds_mcrae_concepts_with_mentions.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-05:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/mention_embeds_mcrae_cons_with_mentions.json

echo 'Job Finished !!!'