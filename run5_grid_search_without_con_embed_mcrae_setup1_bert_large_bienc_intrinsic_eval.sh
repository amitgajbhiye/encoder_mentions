#!/bin/bash --login

#SBATCH --job-name=wck_grid_search_without_con_embed_mcrae_setup1_bert_large_bienc_intrinsic_eval

#SBATCH --output=logs/bienc_intrinsic_eval/out_k_grid_search_without_con_embed_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt
#SBATCH --error=logs/bienc_intrinsic_eval/err_k_grid_search_without_con_embed_mcrae_setup1_bert_large_bienc_intrinsic_eval.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 5-00:00:00
#SBATCH --qos=gpu7d

conda activate venv

python3 src/knn_biencoder_intrinsic_eval.py --config configs/bienc_intrinsic_eval/k_grid_search_without_con_embed_mcrae_setup1_bert_large_bienc_intrinsic_eval.json

echo 'Job Finished !!!'
