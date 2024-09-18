#!/bin/bash --login

#SBATCH --job-name=WN_MenEmbed

#SBATCH --output=logs/get_mention_embeddings/out_get_wordnet_expansion_training_test_mention_embeds.txt
#SBATCH --error=logs/get_mention_embeddings/err_get_wordnet_expansion_training_test_mention_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=25G
#SBATCH -t 0-06:00:00

conda activate venv

python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wordnet_expansion_train_sents.json
python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/wordnet_expansion_trial_sents.json

echo 'Job Finished !!!'