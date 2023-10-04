#!/bin/bash --login

#SBATCH --job-name=wicTSV

#SBATCH --output=logs/wictsv_evaluation/out_testset_multitask_cross_entropy_dist_bert_large_580kdata.txt
#SBATCH --error=logs/wictsv_evaluation/err_testset_multitask_cross_entropy_dist_bert_large_580kdata.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-01:00:00

conda activate venv

CUDA_VISIBLE_DEVICES=0 python3 src/unsup_wictsv_evaluation.py --config configs/wictsv/test_cross_entropy_multitask_580k_bert_large.json

echo 'Job Finished !!!'
