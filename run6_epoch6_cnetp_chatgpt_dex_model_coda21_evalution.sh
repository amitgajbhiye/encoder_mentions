#!/bin/bash --login

#SBATCH --job-name=CnetChatgptEp6_Dexcoda21

#SBATCH --output=logs/coda21_evaluation/out_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.txt
#SBATCH --error=logs/coda21_evaluation/err_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-08:00:00

conda activate venv

python3 src/coda_evaluation.py --config configs/coda21/setup1_epoch6_cnetp_chatgpt_dex_men_encoder.json


echo 'Job Finished !!!'
