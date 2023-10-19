#!/bin/bash --login

#SBATCH --job-name=SupWicTSV

#SBATCH --output=logs/supervised_wictsv/out_setup1_bert_large.txt
#SBATCH --error=logs/supervised_wictsv/err_setup1_bert_large.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=30G
#SBATCH -t 0-06:00:00

## SBATCH --exclusive

conda activate venv

python3 src/supervised_wictsv.py --config configs/supervised_wictsv/setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.json

echo 'Job Finished !!!'
