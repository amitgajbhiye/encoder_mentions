#!/bin/bash --login

#SBATCH --job-name=ccdexSupWicTSV

#SBATCH --output=logs/supervised_wictsv/out_ccdex_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.txt
#SBATCH --error=logs/supervised_wictsv/err_ccdex_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=40G
#SBATCH -t 1-00:00:00

## SBATCH --exclusive

conda activate venv

python3 src/supervised_wictsv.py --config configs/supervised_wictsv/ccdex_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.json

echo 'Job Finished !!!'
