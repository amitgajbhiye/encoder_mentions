#!/bin/bash --login

#SBATCH --job-name=getMenEmb_WN_food

#SBATCH --output=logs/get_mention_embeddings/out_get_wn_food_taxo_mention_embeddings.txt
#SBATCH --error=logs/get_mention_embeddings/err_get_wn_food_taxo_mention_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-05:00:00

conda activate venv

# Getting on wn_food taxonomy embeddings after individually perprocessing it
python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wn_food_taxo.json

# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/environment_ev_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/equipment_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/food_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/science_ev_taxo.json

# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/science_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wn_chemical_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wn_equipment_taxo.json
# python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wn_science_taxo.json


echo 'Job Finished !!!'