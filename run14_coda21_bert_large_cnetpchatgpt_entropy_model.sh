#!/bin/bash --login

#SBATCH --job-name=coda21

#SBATCH --output=logs/coda21_evaluation/out_bert_large_cnepchatgpt_entropy_model.txt
#SBATCH --error=logs/coda21_evaluation/err_bert_large_cnepchatgpt_entropy_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-4:30:00

conda activate venv

python3 src/coda_evaluation.py --config configs/coda21/bert_large_cnepchatgpt_entropy_model.json

echo 'Job Finished !!!'
