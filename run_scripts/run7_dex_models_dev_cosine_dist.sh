#!/bin/bash --login

#SBATCH --job-name=wicTSVDevCosine

#SBATCH --output=logs/wictsv_evaluation/out_run7_dex_models_dev_cosine_dist.txt
#SBATCH --error=logs/wictsv_evaluation/err_run7_dex_models_dev_cosine_dist.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_setup1_dex_men_encoder.json
python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_setup1_epoch2_cnetp_chatgpt_dex_men_encoder.json
python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/dev_cosine_setup1_epoch6_cnetp_chatgpt_dex_men_encoder.json

echo 'Job Finished !!!'
