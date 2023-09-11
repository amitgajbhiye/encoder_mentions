#!/bin/bash --login

#SBATCH --job-name=ccMenEnc

#SBATCH --output=logs/mention_enc/out_conceptcontra_bert_large_cnetpchatgpt_5k_cons.txt
#SBATCH --error=logs/mention_enc/err_conceptcontra_bert_large_cnetpchatgpt_5k_cons.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 2-00:00:00

conda activate venv

python3 src/mention_encoder.py --config_file configs/mention/conceptcontra_bert_large_cnetpchatgpt_5k_cons.json

echo 'Job Finished !!!'
