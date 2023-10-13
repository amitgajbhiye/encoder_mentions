#!/bin/bash --login

#SBATCH --job-name=s2oly_tran_wine_mention_embeds

#SBATCH --output=logs/get_mention_embeddings/out_setup2_oly_transport_wine_mention_embeds.txt
#SBATCH --error=logs/get_mention_embeddings/err_setup2_oly_transport_wine_mention_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-02:30:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/small_onto_mention/setup2_olympics_mention_embeddings.json
python3 src/get_mention_embeddings.py --config_file configs/small_onto_mention/setup2_transport_mention_embeddings.json
python3 src/get_mention_embeddings.py --config_file configs/small_onto_mention/setup2_wine_mention_embeddings.json

echo 'Job Finished !!!'