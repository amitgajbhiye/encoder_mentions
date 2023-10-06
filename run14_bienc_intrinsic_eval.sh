#!/bin/bash --login

#SBATCH --job-name=BiEncInt

#SBATCH --output=logs/bienc_intrinsic_eval/out_bert_large.txt
#SBATCH --error=logs/bienc_intrinsic_eval/err_bert_large.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/biencoder_intrinsic_eval.py --config configs/bienc_intrinsic_eval/bert_large_bienc_intrinsic_eval.json

echo 'Job Finished !!!'
