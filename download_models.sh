#!/bin/bash --login 

mkdir -p logs 
mkdir -p trained_models/embeddings

cd trained_models


echo "Downloading Pretrained Models ..."

wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_definition_encoder_wordnet_codwoe.pt
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_mention_encoder_cnetpchatgpt_5k_cons.pt