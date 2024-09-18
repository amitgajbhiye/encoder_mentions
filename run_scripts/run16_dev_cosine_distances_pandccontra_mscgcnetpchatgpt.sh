#!/bin/bash --login

#SBATCH --job-name=wicTSV

#SBATCH --output=logs/wictsv_evaluation/out_dev_cosine_distances_PandCcontra_mscgcnetpchatgpt.txt
#SBATCH --error=logs/wictsv_evaluation/err_dev_cosine_distances_PandCcontra_mscgcnetpchatgpt.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-00:30:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_dist_conceptcontra_cnetpchatgpt_bert_large.json
python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_dist_mscgcnetchatgpt_cnetpchatgpt_bert_large.json
python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_dist_propertycontra_cnetpchatgpt_bert_large.json

echo 'Job Finished !!!'