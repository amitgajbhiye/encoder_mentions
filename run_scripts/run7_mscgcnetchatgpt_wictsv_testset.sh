#!/bin/bash --login

#SBATCH --job-name=mscgcnetchatgpt_wictsv_testset

#SBATCH --output=logs/wictsv_evaluation/out_mscgcnetchatgpt_testset.txt
#SBATCH --error=logs/wictsv_evaluation/err_mscgcnetchatgpt_testset.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-05:00:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/mscgcnetchatgpt_testset.json

echo 'Job Finished !!!'