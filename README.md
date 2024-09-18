## Mention & Definition Encoders

### Get Mention and Definition Embeddings from Pretrained Models
```

# Install Git LFS (https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

#Clone the repo
git clone git@github.com:amitgajbhiye/encoder_mentions.git

# Download the pretrained mention and definition encoders. 
#The models will be downloaded in trained_models directory.

cd encoder_mentions
sh download_models.sh

# To get mention embeddings run:
python3 src/get_mention_embeddings.py --config_file configs/get_mention_embeddings/get_mention_embeds.json

#To get definition embeddings run:
python3 src/get_definition_embeddings.py --config_file configs/get_definition_embeddings/get_definition_embeds.json

```

The `config_file` has the required parameters to guide the mention and definition embeddings generation. The default configuration file mentioned above for mention and definition generation generates the embeddings for the dummy science taxonomy mentiond in `word_sent_file` from the downloaded models saved in `trained models` directory.

The following are the configuration file parameters:

- dataset_name: The name of the dataset for which embeddings are generated. This will also be the name of the final pickle file of embddings generated in `save_dir` directory of config file.

- hf_tokenizer_name and hf_checkpoint_name:  The name of the huggingface tokenizer and model ids. The default is - `bert-large-uncased`.

- word_sent_file: The file containing input data for which mention or definition embeddings are to be generated. The input file is in the tab seperated format of -  (word sentence).  

- pretrained_model_path: The path of the mention and definition encoders as downloaded above.

- save_dir: The directory path where final embedding files will be saved. The embeddings are saved in a pickled list in the format - (word sentence embedding).

 