#!/bin/bash --login

#SBATCH --job-name=extSent

#SBATCH --output=logs/extract_sents/out_extract_wiki_sents.txt
#SBATCH --error=logs/extract_sents/err_extract_wiki_sents.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH --qos="gpu7d"


##SBATCH -p compute,highmem
#SBATCH --mem=20G
#SBATCH -t 3-00:00:00

conda activate venv

python3 src/extract_sentences.py

echo 'Job Finished !!!'