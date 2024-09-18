#!/bin/bash --login

#SBATCH --job-name=s1CC3dexMention_encoder_sumo_mention_embeddings

#SBATCH --output=logs/get_mention_embeddings/out_setup1_cnetpchatgpt_3dex_mention_encoder_sumo_mention_embeddings.txt
#SBATCH --error=logs/get_mention_embeddings/err_setup1_cnetpchatgpt_3dex_mention_encoder_sumo_mention_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=100G
#SBATCH -t 2-00:00:00
#SBATCH --exclusive

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/cnetpchatgpt_dex_mention_encoder_embeds/setup1_sumo_mention_embeddings.json

echo 'Job Finished !!!'
