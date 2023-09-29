#!/bin/bash --login

#SBATCH --job-name=coda21MultiTask

#SBATCH --output=logs/coda21_evaluation/out_cross_ent_men_def_bert_large_cnetpchatgpt.txt
#SBATCH --error=logs/coda21_evaluation/err_cross_ent_men_def_bert_large_cnetpchatgpt.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-05:00:00

conda activate venv

python3 src/coda_evaluation.py --config configs/coda21/cross_ent_men_def_bert_large_cnetpchatgpt.json

echo 'Job Finished !!!'