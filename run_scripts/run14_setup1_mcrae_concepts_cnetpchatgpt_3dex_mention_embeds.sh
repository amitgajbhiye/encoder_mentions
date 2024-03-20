#!/bin/bash --login

#SBATCH --job-name=s1_mcrae_concepts_cnetpchatgpt_3dex_mention_embeds

#SBATCH --output=logs/get_mention_embeddings/out_setup1_mcrae_concepts_cnetpchatgpt_3dex_mention_embeds.txt
#SBATCH --error=logs/get_mention_embeddings/err_setup1_mcrae_concepts_cnetpchatgpt_3dex_mention_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=100G
#SBATCH -t 0-05:00:00
##SBATCH --exclusive

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/setup1_cnetpchatgpt_dex_mention_embeds_mcrae_cons_with_mentions.json
python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/setup1_dex_mention_embeds_mcrae_cons_with_mentions.json

echo 'Job Finished !!!'
