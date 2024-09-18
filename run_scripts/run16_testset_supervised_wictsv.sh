#!/bin/bash --login

#SBATCH --job-name=TestSupWicTSV

#SBATCH --output=logs/supervised_wictsv/out_testset_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.txt
#SBATCH --error=logs/supervised_wictsv/err_testset_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-00:30:00


conda activate venv

python3 src/test_supervised_wictsv.py --config configs/supervised_wictsv/testset_setup1_cnetpchatgpt_men_encoder_wordnet_codwoe_def_enc.json


echo 'Job Finished !!!'
