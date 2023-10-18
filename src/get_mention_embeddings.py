import logging
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.getcwd())

import warnings
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn


from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
)
from transformers import AutoTokenizer


from je_utils import read_config, set_seed
from torch.utils.data import DataLoader, Dataset, SequentialSampler

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if device == torch.device("cpu"):
    print(
        f"*** CPU is used to get the embedding, might be slow; Use a GPU for faster processing",
        flush=True,
    )
else:
    print(f"GPU is used to get the embedding.", flush=True)


CLASSES = {
    "bert-base-uncased": (BertModel, BertForMaskedLM, BertTokenizer, 103),
    "bert-large-uncased": (
        BertModel,
        BertForMaskedLM,
        BertTokenizer,
        103,
    ),
    "roberta-base": (
        RobertaModel,
        "",
        RobertaTokenizer,
        50264,
    ),
    "roberta-large": (
        RobertaModel,
        "",
        RobertaTokenizer,
        50264,
    ),
    "deberta-v3-large": (
        DebertaV2Model,
        "",
        DebertaV2Tokenizer,
        128000,
    ),
}


def set_logger(config):
    log_file_name = os.path.join(
        "logs",
        config.get("log_dirctory"),
        f"log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )

    print("config.get('experiment_name') :", config.get("experiment_name"), flush=True)
    print("\nlog_file_name :", log_file_name, flush=True)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(name)s : %(levelname)s - %(message)s",
    )


log = logging.getLogger(__name__)


class DatasetConceptSentence(Dataset):
    def __init__(self, concept_sent_file, dataset_params):
        if isinstance(concept_sent_file, pd.DataFrame):
            self.data_df = concept_sent_file
            log.info(
                f"Supplied Concept Sentence File is a Dataframe : {self.data_df.shape}",
            )
            self.data_df = self.data_df.astype(
                {
                    "concept": str,
                    "sent": str,
                },
            )
        elif os.path.isfile(concept_sent_file):
            log.info(f"Supplied Concept Property File is a Path : {concept_sent_file}")
            log.info(f"Loading into Dataframe ... ")
            self.data_df = pd.read_csv(
                concept_sent_file,
                sep="\t",
                header=None,
                names=["concept", "sent"],
                dtype={
                    "concept": str,
                    "sent": str,
                },
            )
            log.info(f"loaded_dataframe: {self.data_df}")
        else:
            raise TypeError(f"Input file type is not correct !!! - {concept_sent_file}")

        log.info("final_input_df")
        log.info(self.data_df.head(n=100))

        self.hf_tokenizer_name = dataset_params["hf_tokenizer_name"]
        self.hf_tokenizer_path = dataset_params.get("hf_tokenizer_path")

        _, _, tokenizer_class, _ = CLASSES[self.hf_tokenizer_name]

        log.info(f"tokenizer_class : {tokenizer_class}")

        if self.hf_tokenizer_path:
            self.tokenizer = tokenizer_class.from_pretrained(self.hf_tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_name)

        self.max_len = dataset_params["max_len"]

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        concept = self.data_df["concept"][idx]
        sent = self.data_df["sent"][idx]

        return {"concept": concept, "sent": sent}

    def mask_word_in_sent(self, search_string, input_string):
        def has_metacharacters(word):
            metacharacters = r"[]().*+?|{}^$\\"
            return [char for char in word if char in metacharacters]

        if has_metacharacters(word=search_string):
            raw_search_string = re.escape(search_string)
        else:
            raw_search_string = r"\b" + search_string + r"\b"

        srch_output = re.search(raw_search_string, input_string, flags=re.IGNORECASE)
        no_match_was_found = srch_output is None

        if no_match_was_found:
            print(flush=True)
            print(f"******* concept_not_found ******", flush=True)
            print(f"concept : {search_string}", flush=True)
            print(f"sentence : {input_string}", flush=True)
            raise Exception("Concept is not in the Sentence")
        else:
            start_idx = srch_output.start()
            end_idx = srch_output.end()
            mask_sent = (
                input_string[:start_idx] + self.mask_token + input_string[end_idx:]
            )
            return mask_sent

    def get_sent_ids(self, batch):
        sents = [
            self.mask_word_in_sent(con, sent)
            for con, sent in zip(batch["concept"], batch["sent"])
        ]
        encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return encoded_dict


class ModelMentionEncoder(nn.Module):
    def __init__(self, model_params):
        super(ModelMentionEncoder, self).__init__()
        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params.get("hf_model_path")
        _, model_class, _, self.mask_token_id = CLASSES[self.hf_checkpoint_name]

        log.info(f"model_class : {model_class}")

        if self.hf_model_path:
            self.encoder = model_class.from_pretrained(
                self.hf_model_path, output_hidden_states=True
            )
        else:
            self.encoder = BertForMaskedLM.from_pretrained(
                self.hf_checkpoint_name, output_hidden_states=True
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        pretrained_con_embeds=None,
        labels=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.hidden_states[-1]

        def get_mask_token_embeddings(last_layer_hidden_states):
            _, mask_token_index = (
                input_ids == torch.tensor(self.mask_token_id)
            ).nonzero(as_tuple=True)
            mask_vectors = torch.vstack(
                [
                    torch.index_select(v, 0, torch.tensor(idx))
                    for v, idx in zip(last_layer_hidden_states, mask_token_index)
                ]
            )
            return mask_vectors

        mask_vectors = get_mask_token_embeddings(last_layer_hidden_states=hidden_states)

        return mask_vectors


def prepare_data_and_models(config):
    ############
    inference_params = config["inference_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    load_pretrained = inference_params["load_pretrained"]
    pretrained_model_path = inference_params["pretrained_model_path"]
    batch_size = inference_params["batch_size"]
    word_sent_file = dataset_params["word_sent_file"]
    num_workers = 4

    assert word_sent_file is not None, "please specify input file"

    con_sent_dataset = DatasetConceptSentence(word_sent_file, dataset_params)
    con_sent_sampler = SequentialSampler(con_sent_dataset)

    con_sent_dataloader = DataLoader(
        con_sent_dataset,
        batch_size=batch_size,
        sampler=con_sent_sampler,
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    # Creating Model
    model = nn.DataParallel(ModelMentionEncoder(model_params=model_params))
    model.to(device=device)

    if load_pretrained:
        log.info(f"load_pretrained is : {load_pretrained}")
        log.info(f"Loading Pretrained Model Weights From : {pretrained_model_path}")

        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

        log.info(f"Loaded Pretrained Model")

        for name, parameter in model.named_parameters():
            log.info(f"layer_name: {name}")

    log.info(f"model_class : {model.__class__.__name__}")

    return {
        "model": model,
        "con_sent_dataset": con_sent_dataset,
        "con_sent_dataloader": con_sent_dataloader,
    }


def run_model(config, param_dict):
    model = param_dict["model"]

    con_sent_dataset = param_dict["con_sent_dataset"]
    con_sent_dataloader = param_dict["con_sent_dataloader"]
    inference_params = config["inference_params"]
    dataset_params = config["dataset_params"]

    con_sent_embed = []
    model.eval()
    for step, batch in enumerate(tqdm(con_sent_dataloader, desc="Iter")):
        log.info(f"Processing batch {step+1} / {len(con_sent_dataloader)}")
        # print(f"concept : {batch['concept']}", flush=True)
        # print(f"sents : {batch['sent']}", flush=True)

        ids_dict = con_sent_dataset.get_sent_ids(batch)
        ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

        with torch.no_grad():
            outputs = model(pretrained_con_embeds=None, **ids_dict)

        mask_vectors = outputs
        mask_vectors = mask_vectors.cpu().numpy()

        for con, sent, embed in zip(batch["concept"], batch["sent"], mask_vectors):
            con_sent_embed.append((con, sent, embed))

    save_dir = inference_params["save_dir"]
    dataset_name = dataset_params["dataset_name"]
    out_file_name = os.path.join(save_dir, f"{dataset_name}.pkl")

    with open(out_file_name, "wb") as pkl_file:
        pickle.dump(con_sent_embed, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

    log.info(f"Mention Embeddings are saved in : {out_file_name}")
    log.info("Program Finished.")


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser(description="Mention Encoder")
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="path to the configuration file",
    )

    args = parser.parse_args()
    config = read_config(args.config_file)

    set_logger(config=config)
    log = logging.getLogger(__name__)
    log.info("The model is run with the following configuration")
    log.info(f"\n {config} \n")

    param_dict = prepare_data_and_models(config=config)
    run_model(config=config, param_dict=param_dict)
