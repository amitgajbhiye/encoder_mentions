import torch
import os
import torch.nn as nn
import logging

import time
import pandas as pd
from scipy.spatial import distance


from get_mention_embeddings import ModelMentionEncoder as mention_encoder
from get_definition_embeddings import ModelDefinitionEncoder as definition_encoder
from torch.utils.data import DataLoader, Dataset

from multitask_con_prop_men_def import JointConceptPropDefMen
from je_utils import read_config, set_seed, compute_scores
from argparse import ArgumentParser

from transformers import (
    AdamW,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


CLASSES = {
    "bert-base-uncased": (BertModel, BertForMaskedLM, BertTokenizer, 103),
    "bert-large-uncased": (
        BertModel,
        BertForMaskedLM,
        BertTokenizer,
        103,
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MentionDefinitionDataset(Dataset):
    def __init__(self, concept_mention_definition_file, dataset_params):
        self.data_df = pd.read_csv(
            concept_mention_definition_file,
            sep="\t",
            header=None,
            names=["concept", "mention", "definition"],
            dtype={"concept": str, "mention": str, "definition": str},
        )
        log.info(f"loaded_dataframe: {self.data_df}")

        print(f"Initial self.data_df", flush=True)
        print(self.data_df, flush=True)

        self.data_df["whole_word_present"] = self.data_df.apply(
            lambda x: self.check_whole_word_in_sent(x.concept, x.mention), axis=1
        )

        no_whole_word_df = self.data_df[self.data_df["whole_word_present"] == "no"]
        print("no_whole_word_df", flush=True)
        print(no_whole_word_df, flush=True)

        self.data_df = self.data_df[self.data_df["whole_word_present"] == "yes"]
        print(f"self.data_df after removing non whole words", flush=True)
        print(self.data_df, flush=True)

        # +++++++++++++++++++++++++++++++++++++++++++++++

        self.data_df = self.data_df.sample(frac=1)
        self.data_df.reset_index(inplace=True, drop=True)
        self.unique_cons = self.data_df["concept"].unique()

        self.data_df["labels"] = 0
        self.data_df.set_index("concept", inplace=True, drop=False)

        for lbl, con in enumerate(self.unique_cons, start=1):
            self.data_df.loc[con, "labels"] = lbl
        self.data_df.reset_index(inplace=True, drop=True)

        self.data_df = self.data_df[["concept", "mention", "definition", "labels"]]

        log.info("final_input_df")
        log.info(self.data_df.head(n=100))

        self.hf_tokenizer_name = dataset_params["hf_tokenizer_name"]
        self.hf_tokenizer_path = dataset_params["hf_tokenizer_path"]

        _, _, tokenizer_class, _ = CLASSES[self.hf_tokenizer_name]

        log.info(f"tokenizer_class : {tokenizer_class}")

        self.tokenizer = tokenizer_class.from_pretrained(self.hf_tokenizer_path)

        self.max_len = dataset_params["max_len"]

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        concept = self.data_df["concept"][idx]
        sent = self.data_df["sent"][idx]
        labels = self.data_df["labels"][idx]

        return {"concept": concept, "sent": sent, "labels": labels}

    def check_whole_word_in_sent(self, concept, sent):
        def has_metacharacters(word):
            metacharacters = r"[]().*+?|{}^$\\"
            return [char for char in word if char in metacharacters]

        if has_metacharacters(word=concept):
            raw_search_string = re.escape(concept)
        else:
            raw_search_string = r"\b" + concept + r"\b"

        srch_output = re.search(raw_search_string, sent, flags=re.IGNORECASE)

        if srch_output:
            return "yes"
        else:
            return "no"

    def mask_word_in_sent(self, search_string, input_string):
        def has_metacharacters(word):
            metacharacters = r"[]().*+?|{}^$\\"
            return [char for char in word if char in metacharacters]

        if has_metacharacters(word=search_string):
            raw_search_string = re.escape(search_string)
        else:
            raw_search_string = r"\b" + search_string + r"\b"

        srch_output = re.search(raw_search_string, input_string, flags=re.IGNORECASE)

        start_idx = srch_output.start()
        end_idx = srch_output.end()
        mask_sent = input_string[:start_idx] + self.mask_token + input_string[end_idx:]

        return mask_sent

    def get_sent_ids(self, batch):
        sents = [
            self.mask_word_in_sent(con, sent)
            for con, sent in zip(batch["concept"], batch["sent"])
        ]

        print(f"masked_sents", flush=True)
        print(sents, flush=True)

        encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        encoded_dict["labels"] = batch["labels"]

        return encoded_dict


class MentionDefinitionEncoder(nn.Module):
    def __init__(self, config):
        super(MentionDefinitionEncoder, self).__init__()

        inference_params = config["inference_params"]
        dataset_params = config["dataset_params"]
        model_params = config["model_params"]

        pretrained_mention_model_path = inference_params[
            "pretrained_mention_model_path"
        ]
        pretrained_definition_model_path = inference_params[
            "pretrained_definition_model_path"
        ]

        # Creating Mention Model
        self.men_model = nn.DataParallel(mention_encoder(model_params=model_params))
        self.men_model.to(device=device)

        # Creating Definition Model
        self.def_model = nn.DataParallel(definition_encoder(model_params=model_params))
        self.def_model.to(device=device)

        self.men_model.load_state_dict(torch.load(pretrained_mention_model_path))
        self.def_model.load_state_dict(torch.load(pretrained_definition_model_path))

        print(
            f"Mention Model is loaded from : {pretrained_mention_model_path}",
            flush=True,
        )
        print(
            f"Definition Model is loaded from : {pretrained_definition_model_path}",
            flush=True,
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            dataset_params["hf_tokenizer_path"]
        )
        self.max_len = dataset_params["max_len"]

    def get_mention_embeds(self, context_sents):
        ids_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=context_sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

        with torch.no_grad():
            mention_vectors = self.men_model(pretrained_con_embeds=None, **ids_dict)

        return mention_vectors

    def get_definition_embeds(self, definition_sents):
        ids_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=definition_sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

        with torch.no_grad():
            def_vectors = self.def_model(pretrained_con_embeds=None, **ids_dict)

        return def_vectors

    def forward(self, context_sents, definition_sents):
        mention_embeds = self.get_mention_embeds(context_sents=context_sents)
        definition_embeds = self.get_definition_embeds(
            definition_sents=definition_sents
        )

        print(f"mention_embeds.shape : {mention_embeds.shape}", flush=True)
        print(f"definition_embeds.shape : {definition_embeds.shape}", flush=True)

        cosine_distances = []
        for men_emb, def_emb in zip(mention_embeds, definition_embeds):
            cos_dist = distance.cosine(men_emb.cpu().numpy(), def_emb.cpu().numpy())
            # print(f"cos_dist: {type(cos_dist)}, {cos_dist}")
            cosine_distances.append(cos_dist.item())

        print(f"cosine_distances : {cosine_distances}", flush=True)

        return cosine_distances


def prepare_data_and_models(config):
    pass


def train():
    pass


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser(description="mention-definition encoder")

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

    train(config=config, param_dict=param_dict)
else:
    log = logging.getLogger(__name__)
