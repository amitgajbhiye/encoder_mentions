#!/bin/bash --login

#SBATCH --job-name=test_setup1_epoch6_cnetp_chatgpt_dex_men_encoder

#SBATCH --output=logs/wictsv_evaluation/out_test_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.txt
#SBATCH --error=logs/wictsv_evaluation/err_test_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-00:30:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/test_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.json

echo 'Job Finished !!!'
