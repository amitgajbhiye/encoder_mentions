#!/bin/bash --login

#SBATCH --job-name=getMenEmb_dummy

#SBATCH --output=logs/get_mention_embeddings/out_get_dummy_mentionm_embeds.txt
#SBATCH --error=logs/get_mention_embeddings/err_get_dummy_mentionm_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/get_mention_embeds.json


echo 'Job Finished !!!'