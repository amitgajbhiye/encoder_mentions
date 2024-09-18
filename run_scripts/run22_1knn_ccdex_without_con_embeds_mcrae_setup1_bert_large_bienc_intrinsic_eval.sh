#!/bin/bash --login

#SBATCH --job-name=ccDexKNN_BiEncInt

#SBATCH --output=logs/bienc_intrinsic_eval/out_1knn_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt
#SBATCH --error=logs/bienc_intrinsic_eval/err_1knn_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 1-00:00:00

conda activate venv

python3 src/knn_biencoder_intrinsic_eval.py --config configs/bienc_intrinsic_eval/1knn_ccdex_without_con_embeds_mcrae_setup1_bert_large_bienc_intrinsic_eval.json

echo 'Job Finished !!!'