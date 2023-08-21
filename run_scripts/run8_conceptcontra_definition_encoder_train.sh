#!/bin/bash --login

#SBATCH --job-name=ccdefEnc

#SBATCH --output=logs/definition_enc/out_conceptcontra_bert_large_wordnet_codwoe.txt
#SBATCH --error=logs/definition_enc/err_conceptcontra_bert_large_wordnet_codwoe.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 2-00:00:00

conda activate venv

python3 src/definition_encoder.py --config_file configs/definition/conceptcontra_bert_large_wordnet_codwoe.json

echo 'Job Finished !!!'