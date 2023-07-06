import torch
import torch.nn as nn
import csv
import numpy as np
from scipy.spatial import distance

import pickle

from get_mention_embeddings import ModelMentionEncoder as mention_encoder
from get_definition_embeddings import ModelDefinitionEncoder as definition_encoder

from transformers import BertTokenizer
from je_utils import read_config, set_seed, compute_scores
from argparse import ArgumentParser


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class UnsupervisedWicTsv(nn.Module):
    def __init__(self, config):
        super(UnsupervisedWicTsv, self).__init__()

        training_params = config["training_params"]
        dataset_params = config["dataset_params"]
        model_params = config["model_params"]

        pretrained_mention_model_path = training_params["pretrained_mention_model_path"]
        pretrained_definition_model_path = training_params[
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

    def get_mention_embeds(self, context_sents):
        max_len = 120

        ids_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=context_sents,
            max_length=max_len,
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
        max_len = 120

        ids_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=definition_sents,
            max_length=max_len,
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


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__ == "__main__":
    set_seed(42)

    parser = ArgumentParser(description="WiCTSV Evaluation")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    print(f"Reading Configuration File: {args.config_file}", flush=True)
    config = read_config(args.config_file)

    print("The model is run with the following configuration", flush=True)
    print(f"\n {config} \n", flush=True)

    inference_params = config["training_params"]

    batch_size = inference_params["batch_size"]
    wictsv_file = inference_params["wictsv_test_file"]

    data = _read_tsv(wictsv_file)[1:]

    print(f"num_test_instance : {len(data)}", flush=True)

    un_wictsv = UnsupervisedWicTsv(config=config)

    all_preds = []
    for batch_no, i in enumerate(range(0, len(data), batch_size)):
        print(flush=True)
        print(
            f"Processing Batch : {batch_no} / {len(data)//batch_size + 1}", flush=True
        )

        words = []
        context_sents = []
        definitions = []

        batch = data[i : i + batch_size]

        for word, _, context, definition, _, _ in batch:
            words.append(word)
            context_sents.append(
                context.lower().replace(word.lower(), un_wictsv.tokenizer.mask_token)
            )
            definitions.append(
                un_wictsv.tokenizer.mask_token + ":" + " " + definition.lower()
            )

        print(f"words : {words}", flush=True)
        print(f"***context_sents : {len(context_sents)}, {context_sents}", flush=True)
        print(flush=True)
        print(f"***definitions : {len(definitions)}, {definitions}", flush=True)

        assert len(context_sents) == len(
            definitions
        ), "In batch context_sents len not equal to definitions."

        cosine_distance = un_wictsv(
            context_sents=context_sents, definition_sents=definitions
        )

        all_preds.extend(list(cosine_distance))

    print(f"all_preds: {len(all_preds)}, {all_preds}", flush=True)

    with open(
        "trained_models/wictsv_dev_cos_dist/wictsv_developmentset_cosine_distances.pickle",
        "wb",
    ) as pkl_file:
        pickle.dump(all_preds, pkl_file)

    # labels = np.array(_read_tsv(input_file=inference_params["label_file"])).flatten()
    # labels = np.array([1 if label == "T" else 0 for label in labels], dtype=int)

    # print(f"labels : {labels}", flush=True)
    # print(f"all_preds : {all_preds}", flush=True)

    # scores = compute_scores(labels=labels, preds=all_preds)

    # print(flush=True)
    # for key, value in scores.items():
    #     print(key, ":", value, flush=True)
