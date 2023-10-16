import torch
import os
import torch.nn as nn
import csv
import numpy as np
import pandas as pd
import logging
import time
from scipy.spatial import distance


import pickle

from get_mention_embeddings import ModelMentionEncoder as mention_encoder
from get_definition_embeddings import ModelDefinitionEncoder as definition_encoder

from multitask_con_prop_men_def import JointConceptPropDefMen

from transformers import BertTokenizer
from je_utils import read_config, set_seed, compute_scores
from argparse import ArgumentParser
from torch.nn import BCEWithLogitsLoss


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


class SupervisedWicTsv(nn.Module):
    def __init__(self, config):
        super(SupervisedWicTsv, self).__init__()

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
        self.classifier = nn.Linear(2 * self.men_model.hidden_size, 1)
        self.dropout = nn.Dropout(2 * self.men_model.hidden_dropout_prob)

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
        def_vectors = self.def_model(pretrained_con_embeds=None, **ids_dict)

        return def_vectors

    def forward(self, context_sents, definition_sents):
        mention_embeds = self.get_mention_embeds(context_sents=context_sents)
        definition_embeds = self.get_definition_embeds(
            definition_sents=definition_sents
        )

        concatenated_embeds = torch.cat((mention_embeds, definition_embeds), dim=1)

        print(f"mention_embeds.shape : {mention_embeds.shape}", flush=True)
        print(f"definition_embeds.shape : {definition_embeds.shape}", flush=True)
        print(f"pooled_embeds.shape : {concatenated_embeds.shape}", flush=True)

        concatenated_embeds = nn.ReLU()(concatenated_embeds)
        concatenated_embeds = self.dropout(concatenated_embeds)
        logits = self.classifier(concatenated_embeds)

        outputs = (logits,)

        if label is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), label)
            outputs = (loss,) + outputs

        return outputs

    ###################################################################


def prepare_data_and_models(config):
    ############
    training_params = config["training_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    load_pretrained = training_params["load_pretrained"]
    pretrained_model_path = training_params["pretrained_model_path"]
    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    max_epochs = training_params["max_epochs"]
    batch_size = training_params["batch_size"]

    train_file = dataset_params["train_file_path"]
    valid_file = dataset_params["val_file_path"]

    num_workers = 4

    if train_file is not None:
        train_dataset = DatasetConceptSentence(train_file, dataset_params)
        train_sampler = MPerClassSampler(
            train_dataset.data_df["labels"],
            m=1,
            batch_size=batch_size,
            length_before_new_iter=len(train_dataset.data_df["labels"]),
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
        )

        log.info(f"Train Data DF shape : {train_dataset.data_df.shape}")
    else:
        log.info(f"Train File is Empty.")
        train_dataloader = None

    if valid_file is not None:
        val_dataset = DatasetConceptSentence(valid_file, dataset_params)
        val_sampler = MPerClassSampler(
            val_dataset.data_df["labels"],
            m=1,
            batch_size=batch_size,
            length_before_new_iter=len(val_dataset.data_df["labels"]),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        log.info(f"Valid Data DF shape : {val_dataset.data_df.shape}")
    else:
        log.info(f"Validation File is Empty.")
        val_dataloader = None

    ########### Creating Model ###########
    model = ModelDefinitionEncoder(model_params=model_params)

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")
    if load_pretrained:
        log.info(f"load_pretrained is : {load_pretrained}")
        log.info(f"Loading Pretrained Model Weights From : {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

        log.info(f"Loaded Pretrained Model")

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info(f"using multiple GPUs: {n_gpu}")
            model = nn.DataParallel(model)
        model.to(device=device)

    log.info(f"model_class : {model.__class__.__name__}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if training_params["lr_schedule"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            len(train_dataloader) * 2,
            int(len(train_dataloader) * max_epochs),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            len(train_dataloader) * 2,
            int(len(train_dataloader) * max_epochs),
            num_cycles=1.5,
        )
    log.info(f"lr_scheduler: {scheduler}")

    return {
        "model": model,
        "scheduler": scheduler,
        "optimizer": optimizer,
        "train_dataset": train_dataset,
        "train_dataloader": train_dataloader,
        "val_dataset": val_dataset,
        "val_dataloader": val_dataloader,
    }


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser(description="Definition Encoder")

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


class MultitaskSupervisedWicTsv(nn.Module):
    def __init__(self, config):
        super(MultitaskSupervisedWicTsv, self).__init__()

        inference_params = config["inference_params"]
        dataset_params = config["dataset_params"]
        model_params = config["model_params"]

        pretrained_multitask_model_path = inference_params[
            "pretrained_multitask_model_path"
        ]

        # Creating and loading Multitask Model
        self.multitask_model = JointConceptPropDefMen(model_params=model_params)
        self.multitask_model.to(device=device)
        self.multitask_model.load_state_dict(
            torch.load(pretrained_multitask_model_path)
        )

        print(
            f"Multitask Model is loaded from : {pretrained_multitask_model_path}",
            flush=True,
        )
        print(
            f"multitask_model_class: {self.multitask_model.__class__.__name__}",
            flush=True,
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            dataset_params["hf_tokenizer_path"]
        )
        self.max_len = dataset_params["max_len"]

    def multitask_get_context_mention_embeds(self, context_sents, definition_sents):
        mention_encodings = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=context_sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        definition_encodings = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=definition_sents,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        mention_encodings = {
            key: value.to(device) for key, value in mention_encodings.items()
        }
        definition_encodings = {
            key: value.to(device) for key, value in definition_encodings.items()
        }

        input_ids_and_labels = {
            "con_prop_encodings": None,
            "property_encodings": None,
            "con_prop_labels": None,
            ###
            "con_def_encodings": None,
            "definition_encodings": definition_encodings,
            "con_defs_labels": None,
            ###
            "con_mens_encodings": None,
            "mention_encodings": mention_encodings,
            "con_mens_labels": None,
        }

        ### Multitask Model

        outputs = self.multitask_model(input_ids_and_labels)

        mention_vectors = outputs["men_masks"]
        def_vectors = outputs["def_masks"]

        return mention_vectors, def_vectors

    def forward(self, context_sents, definition_sents):
        mention_embeds, definition_embeds = self.multitask_get_context_mention_embeds(
            context_sents=context_sents, definition_sents=definition_sents
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
    """Reads a tab separated file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__ == "__main__":
    set_seed(1)

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

    model_type = config["inference_params"]["model_type"]
    ######## Creating Model ########
    if model_type == "multitask":
        un_wictsv = MultitaskUnsupervisedWicTsv(config=config)
    else:
        un_wictsv = UnsupervisedWicTsv(config=config)

    inference_params = config["inference_params"]
    batch_size = inference_params["batch_size"]
    wictsv_file = inference_params["wictsv_test_file"]

    data = _read_tsv(wictsv_file)[1:]  # word idx context definition domain hypernym
    data_df = pd.DataFrame.from_records(data)
    data_df.rename(
        columns={
            0: "word",
            1: "idx",
            2: "context",
            3: "definition",
            4: "domain",
            5: "hypernym",
        },
        inplace=True,
    )

    # Reading Labels
    if inference_params["label_file"]:
        label = np.array(_read_tsv(inference_params["label_file"])).flatten()
        label = np.array([1 if l == "T" else 0 for l in label], dtype=int)

        print(f"label : {label}", flush=True)

        assert (
            data_df.shape[0] == label.shape[0]
        ), "Number of input records is not equal to labels"

        data_df["label"] = label

    print(f"data_df.columns : {data_df.columns}", flush=True)
    print(f"data_df : {data_df}", flush=True)

    # test_domain: '0': 717, - WNT/WKT # -1 in configfile
    # test_domain: '1': 205, - MSH
    # test_domain: '2': 216, - CTL
    # test_domain: '3': 168, - CPS

    data_type = config["dataset_params"]["dataset_name"]

    if data_type == "wictsv_devset":
        # For getting classification thresholds form dev data.
        test_domains = [("4", "all")]

    else:
        test_domains = [
            ("0", "WNT_WKT"),
            ("1", "MSH"),
            ("2", "CTL"),
            ("3", "CPS"),
            ("4", "all"),
        ]

    for domain_id, domain in test_domains:
        print(
            f"******* Testing on domain_id: {domain_id}, domain: {domain} *******",
            flush=True,
        )

        if domain_id in ("0", "1", "2", "3"):
            domain_data_df = data_df[data_df["domain"] == domain_id]
        else:
            print(f"*** Testing on All Domains ***")
            domain_data_df = data_df

        print(f"num_test_instance : {len(domain_data_df)}", flush=True)

        all_preds, all_labels = [], []

        for batch_no, i in enumerate(range(0, len(domain_data_df), batch_size)):
            print(flush=True)
            print(
                f"Processing Batch : {batch_no} / {len(domain_data_df) // batch_size + 1}",
                flush=True,
            )

            words, context_sents, definitions, batch_labels = (
                [],
                [],
                [],
                [],
            )

            batch = domain_data_df.values[i : i + batch_size]

            for word, _, context, definition, _, _, label in batch:
                words.append(word)
                context_sents.append(
                    context.lower().replace(
                        word.lower(), un_wictsv.tokenizer.mask_token
                    )
                )
                definitions.append(
                    un_wictsv.tokenizer.mask_token + ":" + " " + definition.lower()
                )
                batch_labels.append(label)

            print(f"***words : {len(words)}, {words}", flush=True)
            print(flush=True)
            print(
                f"***context_sents : {len(context_sents)}, {context_sents}", flush=True
            )
            print(flush=True)
            print(f"***definitions : {len(definitions)}, {definitions}", flush=True)
            print(f"***batch_labels : {len(batch_labels)}, {batch_labels}", flush=True)

            assert len(context_sents) == len(
                definitions
            ), "In batch context_sents len not equal to definitions."

            cosine_distance = un_wictsv(
                context_sents=context_sents, definition_sents=definitions
            )

            all_preds.extend(cosine_distance)
            all_labels.extend(batch_labels)

        probs_pkl_file = os.path.join(
            inference_params["save_dir"],
            f"{config['experiment_name']}_{domain_id}_{domain}.pkl",
        )

        with open(probs_pkl_file, "wb") as pkl_file:
            pickle.dump(all_preds, pkl_file)

        if data_type == "wictsv_devset":
            break

        def to_labels(probs, threshold):
            return (probs <= threshold).astype("int")

        classification_thresh = inference_params["classification_thresh"]

        assert (
            classification_thresh is not None
        ), f"Specify classification_thresh. It is: {classification_thresh} now."

        all_preds = to_labels(
            probs=np.array(all_preds), threshold=classification_thresh
        )
        scores = compute_scores(labels=np.array(all_labels), preds=all_preds)

        print(f"all_labels: {len(all_labels)}, {all_labels}", flush=True)
        print(f"all_preds: {len(all_preds)}, {all_preds}", flush=True)

        print(flush=True)
        print(
            f"******* Results on domain_id: {domain_id}, domain: {domain} records: {domain_data_df.shape[0]}*******",
            flush=True,
        )
        for key, value in scores.items():
            print(key, ":", value, flush=True)
        print(flush=True)
