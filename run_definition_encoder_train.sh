#!/bin/bash --login

#SBATCH --job-name=defEnc

#SBATCH --output=logs/definition_enc/out_bert_large_wordnet_codwoe_data.txt
#SBATCH --error=logs/definition_enc/err_bert_large_wordnet_codwoe_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 1-12:00:00

conda activate venv

python3 src/definition_encoder.py --config_file configs/definition/bert_large_wordnet_codwoe.json

echo 'Job Finished !!!'