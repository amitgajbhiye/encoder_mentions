#!/bin/bash --login

#SBATCH --job-name=MultiTask

#SBATCH --output=logs/multitask_encoder_pretrain/out_bert_large_multitask_short_defs_pretrain.txt
#SBATCH --error=logs/multitask_encoder_pretrain/err_bert_large_multitask_short_defs_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mem=60G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/multitask_con_prop_men_def.py --config_file configs/multitask_con_prop_def_men/bert_large_multitask_short_defs_pretrain.json

echo 'Job Finished !!!'