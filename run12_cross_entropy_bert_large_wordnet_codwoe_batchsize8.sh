#!/bin/bash --login

#SBATCH --job-name=EntDefEncBs8

#SBATCH --output=logs/definition_enc/out_cross_entropy_bert_large_wordnet_codwoe_batchsize8.txt
#SBATCH --error=logs/definition_enc/err_cross_entropy_bert_large_wordnet_codwoe_batchsize8.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 1-12:00:00

conda activate venv

python3 src/definition_encoder.py --config_file configs/definition/cross_entropy_bert_large_wordnet_codwoe_batchsize8.json

echo 'Job Finished !!!'
