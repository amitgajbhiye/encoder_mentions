#!/bin/bash --login

#SBATCH --job-name=5extSent

#SBATCH --output=logs/extract_sents/out_extract_sentences_5.txt
#SBATCH --error=logs/extract_sents/err_extract_sentences_5.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem
#SBATCH --mem=20G
#SBATCH -t 1-06:00:00

conda activate venv

python3 src/extract_sentences.py 4000 5500

echo 'Job Finished !!!'