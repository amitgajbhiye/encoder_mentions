#!/bin/bash --login 

mkdir -p logs
mkdir -p trained_models

cd trained_models
mkdir -p setup_1 setup_2 setup_3 setup_4


echo "Downloading Pretrained Models ..."

### SetUp 1 ###
cd setup_1

# 1. BERT-large Concept Property BiEncoder Pretrained on ConceptNet_ChatGPT Data - Cross Entropy Model
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bienc_concept_property_bert_large_cnetp_chatgpt100k_pretrained.pt 

# 1. BERT-large Mention Encoder - Contrastive Model. Concept Embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Cross Entropy Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_mention_encoder_cnetpchatgpt_5k_cons.pt

# 1. BERT-large Definition Encoder - Contrastive Model. Concept Embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Cross Entropy Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bert_large_definition_encoder_wordnet_codwoe.pt

printf 'Finished Downloading SetUp 1 Models' 
# ****************************************

### SetUp 2 ### 
cd ../setup_2
# 2. BERT-large Concept Property BiEncoder Pretrained on MSCG_ConceptNet_ChatGPT Data - Cross Entropy Model
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/bienc_concept_property_bert_large_mscg_cnetp_chatgpt100k_pretrained.pt 

# 2. BERT-large Definition Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on MSCG_ConceptNet_ChatGPT Data Cross Entropy Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/definition_enc_mscg_cnetp_chatgpt100k_entropy_pretrained_model_wordnet_codwoe_trained.pt  

# 2. BERT-large Mention Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on MSCG_ConceptNet_ChatGPT Data Cross Entropy Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/mention_enc_mscg_cnetp_chatgpt100k_entropy_pretrained_model_wiki_mentions_trained.pt 

printf 'Finished Downloading SetUp 2 Models' 
# ****************************************

### SetUp 3 ### 
cd ../setup_3

# 3. BERT-large Concept Property BiEncoder Pretrained on ConceptNet_ChatGPT Data - Concept Centric Contrastive Model
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_concept_centric_bienc_concept_property_bert_large_cnetp_chatgpt100k_pretrained.pt 

# 3. BERT-large Definition Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Concept Centric Contrastive Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_concept_centric_definition_enc_cnetp_chatgpt100k_pretrained_model_wordnet_codwoe_trained.pt 

# 3. BERT-large Mention Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Concept Centric Contrastive Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_concept_centric_mention_enc_cnetp_chatgpt100k_pretrained_model_wiki_mention_trained.pt 


printf 'Finished Downloading SetUp 3 Models'
# ****************************************

### SetUp 4 ###
cd ../setup_4

# 4. BERT-large Concept Property BiEncoder Pretrained on ConceptNet_ChatGPT Data - Property Centric Contrastive Model
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_property_centric_bienc_concept_property_bert_large_cnetp_chatgpt100k_pretrained.pt

# 4. BERT-large Definition Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Property Centric Contrastive Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_property_centric_definition_enc_cnetp_chatgpt100k_pretrained_model_wordnet_codwoe_trained.pt 

# 4. BERT-large Mention Encoder - Contrastive Model. Concept embeddings comes from BiEncoder Pretrained on ConceptNet_ChatGPT Data Property Centric Contrastive Model.
wget https://huggingface.co/amitgajbhiye/mention_and_definition_encoders/resolve/main/contrastive_property_centric_mention_enc_cnetp_chatgpt100k_pretrained_model_wiki_mention_trained.pt 
