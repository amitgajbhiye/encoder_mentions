#!/bin/bash --login

#SBATCH --job-name=getDummyDefinitionEmbeddings

#SBATCH --output=logs/out_get_dummy_definition_embeddings.txt
#SBATCH --error=logs/err_get_dummy_definition_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/get_definition_embeddings.py --config_file configs/get_definition_embeddings/get_definition_embeds.json

echo 'Job Finished !!!'