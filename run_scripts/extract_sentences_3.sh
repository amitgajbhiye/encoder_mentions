#!/bin/bash --login

#SBATCH --job-name=3extSent

#SBATCH --output=logs/extract_sents/out_extract_sentences_3.txt
#SBATCH --error=logs/extract_sents/err_extract_sentences_3.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem
#SBATCH --mem=20G
#SBATCH -t 1-06:00:00

conda activate venv

python3 src/extract_sentences.py 2000 3000

echo 'Job Finished !!!'