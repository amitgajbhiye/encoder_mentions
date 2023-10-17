#!/bin/bash --login

#SBATCH --job-name=MenDex

#SBATCH --output=logs/mention_enc/out_contrastive_bert_large_dex_mentions_pretrained_mention_encoder.txt
#SBATCH --error=logs/mention_enc/err_contrastive_bert_large_dex_mentions_pretrained_mention_encoder.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH --exclusive

conda activate venv

python3 src/mention_encoder.py --config_file configs/mention/contrastive_bert_large_dex_mentions_pretrained_mention_encoder.json

echo 'Job Finished !!!'
