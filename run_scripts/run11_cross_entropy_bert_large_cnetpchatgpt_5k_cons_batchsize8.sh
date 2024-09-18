#!/bin/bash --login

#SBATCH --job-name=EntMenEncBs8

#SBATCH --output=logs/mention_enc/out_cross_entropy_bert_large_cnetpchatgpt_5k_cons_batchsize8.txt
#SBATCH --error=logs/mention_enc/err_cross_entropy_bert_large_cnetpchatgpt_5k_cons_batchsize8.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 2-00:00:00

conda activate venv

python3 src/mention_encoder.py --config_file configs/mention/cross_entropy_bert_large_cnetpchatgpt_5k_cons_batchsize8.json

echo 'Job Finished !!!'
