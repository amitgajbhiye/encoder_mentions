#!/bin/bash --login 

mkdir -p logs 
mkdir -p trained_models

cd trained_models


echo "Downloading Pretrained Models ..."

#BERT-large mention encoder
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_mention_encoder_cnetpchatgpt_5k_cons.pt

## BERT-large Definition Encoder
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_definition_encoder_wordnet_codwoe.pt
