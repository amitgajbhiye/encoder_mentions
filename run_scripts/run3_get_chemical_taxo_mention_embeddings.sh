#!/bin/bash --login

#SBATCH --job-name=chemical_taxo_men_embed

#SBATCH --output=logs/get_mention_embeddings/out_get_chemical_taxo_mention_embeds.txt
#SBATCH --error=logs/get_mention_embeddings/err_get_chemical_taxo_mention_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-08:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/chemical_taxo.json

echo 'Job Finished !!!'