import gc
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
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from early_stop import EarlyStopping
from je_utils import calculate_inbatch_cross_entropy_loss, read_config, set_seed

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

        print(f"Initial self.data_df", flush=True)
        print(self.data_df, flush=True)

        self.data_df["whole_word_present"] = self.data_df.apply(
            lambda x: self.check_whole_word_in_sent(x.concept, x.sent), axis=1
        )

        no_whole_word_df = self.data_df[self.data_df["whole_word_present"] == "no"]
        print("no_whole_word_df", flush=True)
        print(no_whole_word_df, flush=True)

        self.data_df = self.data_df[self.data_df["whole_word_present"] == "yes"]
        print(f"self.data_df after removing non whole words", flush=True)
        print(self.data_df, flush=True)

        self.data_df = self.data_df.sample(frac=1)
        self.data_df.reset_index(inplace=True, drop=True)
        self.unique_cons = self.data_df["concept"].unique()

        self.data_df["labels"] = 0
        self.data_df.set_index("concept", inplace=True, drop=False)

        for lbl, con in enumerate(self.unique_cons, start=1):
            self.data_df.loc[con, "labels"] = lbl
        self.data_df.reset_index(inplace=True, drop=True)

        self.data_df = self.data_df[["concept", "sent", "labels"]]

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


class ModelMentionEncoder(nn.Module):
    def __init__(self, model_params):
        super(ModelMentionEncoder, self).__init__()

        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params["hf_model_path"]

        _, model_class, _, self.mask_token_id = CLASSES[self.hf_checkpoint_name]
        self.encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )

        self.run_mode = model_params["run_mode"]
        self.loss_type = model_params["loss_type"]

        assert self.run_mode in (
            "train",
            "test",
            "inference",
        ), f"Wrong run_mode: {self.run_mode}"

        if (self.run_mode == "train") and (self.loss_type == "contrastive"):
            self.miner = miners.MultiSimilarityMiner()
            self.use_hard_pair = model_params["use_hard_pair"]
            self.contrastive_loss_fn = losses.NTXentLoss(
                temperature=model_params["tau"]
            )
            log.info(f"loss_function: {self.contrastive_loss_fn}")

        elif (self.run_mode == "train") and (self.loss_type == "cross_entropy"):
            self.cross_entropy_loss = nn.BCEWithLogitsLoss()
            log.info(f"loss_function: {self.cross_entropy_loss}")

        log.info(f"log.info: {self.run_mode}")
        log.info(f"model_class : {model_class}")
        log.info(f"loss_type: {self.loss_type}")

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

        print(f"hidden_states : {hidden_states.shape}", flush=True)

        def get_mask_token_embeddings(last_layer_hidden_states):
            _, mask_token_index = (
                input_ids == torch.tensor(self.mask_token_id)
            ).nonzero(as_tuple=True)

            print(f"mask_token_index: {mask_token_index}", flush=True)

            mask_vectors = torch.vstack(
                [
                    torch.index_select(v, 0, torch.tensor(idx))
                    for v, idx in zip(last_layer_hidden_states, mask_token_index)
                ]
            )
            return mask_vectors

        mask_vectors = get_mask_token_embeddings(last_layer_hidden_states=hidden_states)

        loss = None
        if (self.run_mode == "train") and (self.loss_type == "contrastive"):
            emb_all = torch.cat([mask_vectors, pretrained_con_embeds], dim=0)
            print(f"emb_all :{emb_all.shape}", flush=True)

            labels = torch.cat([labels, labels], dim=0)
            print(f"labels :{labels.shape}: {labels}", flush=True, end="\n")

            if self.use_hard_pair:
                hard_pairs = self.miner(emb_all, labels)
                loss = self.contrastive_loss_fn(emb_all, labels, hard_pairs)
            else:
                loss = self.contrastive_loss_fn(emb_all, labels)

        elif (self.run_mode == "train") and (self.loss_type == "cross_entropy"):
            loss, _, _ = calculate_inbatch_cross_entropy_loss(
                pretrained_con_embeds, mask_vectors, self.cross_entropy_loss, device
            )

        return loss, mask_vectors


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
    model = ModelMentionEncoder(model_params=model_params)

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
            logging.info(f"using_multiple_GPUs: {n_gpu}")
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


def train(config, param_dict):
    model = param_dict["model"]
    scheduler = param_dict["scheduler"]
    optimizer = param_dict["optimizer"]

    train_dataset = param_dict["train_dataset"]
    train_dataloader = param_dict["train_dataloader"]

    val_dataset = param_dict["val_dataset"]
    val_dataloader = param_dict["val_dataloader"]

    training_params = config["training_params"]
    max_epochs = training_params["max_epochs"]
    model_name = training_params["model_name"]
    save_dir = training_params["save_dir"]
    patience_early_stopping = training_params["patience_early_stopping"]
    model_file = os.path.join(save_dir, model_name)

    early_stopping = EarlyStopping(
        patience=patience_early_stopping, verbose=True, path=model_file, delta=1e-10
    )

    log.info(f"model_name_file : {model_file}")

    pretrained_con_embeds_path = training_params["pretrained_con_embeds_path"]
    with open(pretrained_con_embeds_path, "rb") as embed_pkl:
        pretrained_con_embeds_dict = pickle.load(embed_pkl)

    for epoch in trange(max_epochs, desc="Epoch"):
        log.info("Epoch {:} of {:}".format(epoch, max_epochs))
        train_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            print(flush=True)
            print(
                f"batch['concept']: {len(batch['concept'])}, {batch['concept']}",
                flush=True,
            )
            print(f"batch['sent']: {len(batch['sent'])}, {batch['sent']}", flush=True)

            pretrained_con_embeds = torch.tensor(
                [pretrained_con_embeds_dict[con] for con in batch["concept"]]
            ).to(device)

            print(f"pretrained_con_embeds.shape: {pretrained_con_embeds.shape}")

            ids_dict = train_dataset.get_sent_ids(batch)
            ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

            model.zero_grad()
            outputs = model(pretrained_con_embeds=pretrained_con_embeds, **ids_dict)
            loss, _ = outputs

            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 100 == 0:
                log.info(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

            del ids_dict
            del pretrained_con_embeds
            del loss
            gc.collect()

        log.info(f"train_step: {step}")
        train_loss /= step + 1

        del step
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        for step, batch in enumerate(tqdm(val_dataloader, desc="val")):
            pretrained_con_embeds = torch.tensor(
                [pretrained_con_embeds_dict[con] for con in batch["concept"]]
            ).to(device)

            ids_dict = val_dataset.get_sent_ids(batch)
            ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

            with torch.no_grad():
                outputs = model(pretrained_con_embeds=pretrained_con_embeds, **ids_dict)

            loss, _ = outputs

            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
            val_loss += loss.item()

            del ids_dict
            del loss
            gc.collect()

        log.info(f"val_step: {step}")
        val_loss /= step + 1

        del step

        log.info(
            "Epoch: %d | train loss: %.4f | val loss: %.4f ",
            epoch + 1,
            train_loss,
            val_loss,
        )

        torch.cuda.empty_cache()

        if epoch >= 1:
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            log.info("Early stopping. Model trained")
            log.info(f"Trained Model is saved at ; {model_file}")
            break

    torch.cuda.empty_cache()


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

    train(config=config, param_dict=param_dict)
else:
    log = logging.getLogger(__name__)
