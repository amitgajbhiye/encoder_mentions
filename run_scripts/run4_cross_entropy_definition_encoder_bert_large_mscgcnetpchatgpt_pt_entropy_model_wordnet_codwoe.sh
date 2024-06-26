#!/bin/bash --login

#SBATCH --job-name=defMSCGent

#SBATCH --output=logs/definition_enc/out_cross_entropy_definition_encoder_bert_large_mscgcnetpchatgpt_pt_entropy_model_wordnet_codwoe.txt
#SBATCH --error=logs/definition_enc/err_cross_entropy_definition_encoder_bert_large_mscgcnetpchatgpt_pt_entropy_model_wordnet_codwoe.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:2

#SBATCH --mem=16G
#SBATCH -t 1-12:00:00

conda activate venv

python3 src/definition_encoder.py --config_file configs/definition/cross_entropy_bert_large_mscgcnetpchatgpt_pt_entropy_model_wordnet_codwoe.json

echo 'Job Finished !!!'
