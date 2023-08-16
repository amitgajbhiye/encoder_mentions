#!/bin/bash --login

#SBATCH --job-name=wicTSVDevCosDist

#SBATCH --output=logs/wictsv_evaluation/out_conceptcontra_dev_cosine_distances.txt
#SBATCH --error=logs/wictsv_evaluation/err_conceptcontra_dev_cosine_distances.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/conceptcontra_dev_cosine_distances.json

echo 'Job Finished !!!'