#!/bin/bash --login

#SBATCH --job-name=tMenEnc

#SBATCH --output=logs/mention_enc/out_dummy_train_model.txt
#SBATCH --error=logs/mention_enc/err_dummy_train_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 0-05:00:00

conda activate venv

python3 src/mention_encoder.py --config_file configs/mention/bert_large.json

echo 'Job Finished !!!'