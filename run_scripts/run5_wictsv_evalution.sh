#!/bin/bash --login

#SBATCH --job-name=wicTSV

#SBATCH --output=logs/wictsv_evaluation/out_all_domain_bert_large_uncased.txt
#SBATCH --error=logs/wictsv_evaluation/err_all_domain_bert_large_uncased.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/wictsv.json

echo 'Job Finished !!!'