#!/bin/bash --login

#SBATCH --job-name=trainModel

#SBATCH --output=logs/train_on_con_embeds/out_dummy_train_model.txt
#SBATCH --error=logs/train_on_con_embeds/err_dummy_train_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=12G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/model.py --config_file configs/dummy.json

echo 'Job Finished !!!'