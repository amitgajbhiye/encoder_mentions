#!/bin/bash --login

#SBATCH --job-name=ccDEXall_BiEncInt

#SBATCH --output=logs/bienc_intrinsic_eval/out_all_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt
#SBATCH --error=logs/bienc_intrinsic_eval/err_all_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-10:00:00

conda activate venv

python3 src/knn_biencoder_intrinsic_eval.py --config configs/bienc_intrinsic_eval/all_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.json

echo 'Job Finished !!!'
