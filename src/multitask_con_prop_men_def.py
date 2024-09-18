import gc
import logging
import os
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
    AutoTokenizer,
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


class DatasetConceptPropDefMen(Dataset):
    def __init__(self, con_prop_men_def_file, dataset_params):
        if isinstance(con_prop_men_def_file, pd.DataFrame):
            self.data_df = con_prop_men_def_file
            log.info(
                f"Supplied Concept Property Mention Definition File is a Dataframe : {self.data_df.shape}",
            )
            self.data_df = self.data_df.astype(
                dtype={
                    "concept": str,
                    "sent2": str,
                    "sent2_type": str,
                    "sent2_len": str,
                },
            )
        elif os.path.isfile(con_prop_men_def_file):
            log.info(
                f"Supplied Concept Property Mention Definition is a Path : {con_prop_men_def_file}"
            )
            log.info(f"Loading into Dataframe ... ")
            self.data_df = pd.read_csv(
                con_prop_men_def_file,
                sep="\t",
                header=None,
                names=["concept", "sent2", "sent2_type", "sent2_len"],
                dtype={
                    "concept": str,
                    "sent2": str,
                    "sent2_type": str,
                    "sent2_len": str,
                },
            )
            log.info(f"loaded_dataframe: {self.data_df}")
        else:
            raise TypeError(
                f"Input file type is not correct !!! - {con_prop_men_def_file}"
            )

        # self.data_df = self.data_df.sample(frac=0.01)  #############################
        self.data_df.reset_index(inplace=True, drop=True)
        self.unique_cons = self.data_df["concept"].unique()

        self.data_df["labels"] = 0
        self.data_df.set_index("concept", inplace=True, drop=False)

        for lbl, con in enumerate(self.unique_cons, start=1):
            self.data_df.loc[con, "labels"] = lbl
        self.data_df.reset_index(inplace=True, drop=True)

        log.info("final_input_df")
        log.info(self.data_df.head(n=100))

        print(self.data_df.head(n=100))

        self.hf_tokenizer_name = dataset_params["hf_tokenizer_name"]
        _, _, tokenizer_class, _ = CLASSES[self.hf_tokenizer_name]

        if dataset_params["hf_tokenizer_path"]:
            self.hf_tokenizer_path = dataset_params["hf_tokenizer_path"]
            self.tokenizer = tokenizer_class.from_pretrained(self.hf_tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_name)

        log.info(f"tokenizer_class : {tokenizer_class}")

        self.max_len = dataset_params["max_len"]

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        concept = self.data_df["concept"][idx]
        sent2 = self.data_df["sent2"][idx]
        sent2_type = self.data_df["sent2_type"][idx]

        labels = self.data_df["labels"][idx]

        concept_prompt = concept + " means " + self.mask_token

        if sent2_type == "property":
            sent2_prompt = sent2 + " means " + self.mask_token
        elif sent2_type == "definition":
            sent2_prompt = self.mask_token + ": " + sent2
        elif sent2_type == "mention":
            sent2_prompt = self.mask_word_in_sent(concept, sent2)

        return {
            "concept_prompt": concept_prompt,
            "sent2_prompt": sent2_prompt,
            "sent2_type": sent2_type,
            "labels": labels,
        }

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

    def get_ids(self, batch):
        def get_encodings(sents):
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

        concept_prompt = batch["concept_prompt"]
        sent2_prompt = batch["sent2_prompt"]
        sent2_type = batch["sent2_type"]
        labels = batch["labels"]

        props, defs, mens = [], [], []
        con_prop, con_defs, con_mens = [], [], []
        con_prop_labels, con_defs_labels, con_mens_labels = [], [], []

        for idx, (con, sent, typ, lbl) in enumerate(
            zip(concept_prompt, sent2_prompt, sent2_type, labels)
        ):
            if typ == "property":
                props.append(sent)
                con_prop.append(con)
                con_prop_labels.append(lbl)

            elif typ == "definition":
                defs.append(sent)
                con_defs.append(con)
                con_defs_labels.append(lbl)

            elif typ == "mention":
                mens.append(sent)
                con_mens.append(con)
                con_mens_labels.append(lbl)
            else:
                raise Exception(f"Sent2 type: {typ}, is incorrect.")

        print(f"in_batch_data", flush=True)
        print(
            f"con_props- len: {len(con_prop)}, cons:{con_prop}, props:{props}, con_prop_labels:{con_prop_labels}",
            flush=True,
        )
        print(flush=True)
        print(
            f"con_defs- len:{len(con_defs)}, cons:{con_defs}, defs:{defs}, con_defs_labels:{con_defs_labels}",
            flush=True,
        )
        print(flush=True)
        print(
            f"con_mens: len:{len(con_mens)},  cons:{con_mens}, mens:{mens}, con_mens_labels:{con_mens_labels}",
            flush=True,
        )
        print(flush=True)

        assert (
            len(con_prop) == len(props) == len(con_prop_labels)
        ), f"#num of concept property data in batch not equal"

        assert (
            len(con_defs) == len(defs) == len(con_defs_labels)
        ), f"#num of concept definition data in batch not equal"

        assert (
            len(con_mens) == len(mens) == len(con_mens_labels)
        ), f"#num of concept property data in batch not equal"

        if props:
            property_encodings = get_encodings(sents=props)
            con_prop_encodings = get_encodings(sents=con_prop)

            property_encodings = {
                key: value.to(device) for key, value in property_encodings.items()
            }
            con_prop_encodings = {
                key: value.to(device) for key, value in con_prop_encodings.items()
            }

            print(
                f"con_prop_encodings: {len(con_prop)}, {con_prop_encodings['input_ids'].shape}"
            )
            print(
                f"property_encodings: {len(props)}, {property_encodings['input_ids'].shape}"
            )
        else:
            property_encodings = None
            con_prop_encodings = None
            con_prop_labels = None

            print(f"con_prop_encodings: {len(con_prop)}")
            print(f"property_encodings: {len(props)}")

        if defs:
            definition_encodings = get_encodings(sents=defs)
            con_def_encodings = get_encodings(sents=con_defs)

            definition_encodings = {
                key: value.to(device) for key, value in definition_encodings.items()
            }
            con_def_encodings = {
                key: value.to(device) for key, value in con_def_encodings.items()
            }

            print(
                f"con_def_encodings: {len(con_defs)}, {con_def_encodings['input_ids'].shape}"
            )
            print(
                f"definition_encodings: {len(defs)}, {definition_encodings['input_ids'].shape}"
            )
        else:
            definition_encodings = None
            con_def_encodings = None
            con_defs_labels = None
            print(f"con_def_encodings: {len(con_defs)}")
            print(f"definition_encodings: {len(defs)}")

        if mens:
            mention_encodings = get_encodings(sents=mens)
            con_mens_encodings = get_encodings(sents=con_mens)

            mention_encodings = {
                key: value.to(device) for key, value in mention_encodings.items()
            }
            con_mens_encodings = {
                key: value.to(device) for key, value in con_mens_encodings.items()
            }

            print(
                f"con_mens_encodings: {len(con_mens)}, {con_mens_encodings['input_ids'].shape}"
            )
            print(
                f"mention_encodings: {len(mens)}, {mention_encodings['input_ids'].shape}"
            )
        else:
            mention_encodings = None
            con_mens_encodings = None
            con_mens_labels = None
            print(f"con_mens_encodings: {len(con_mens)}")
            print(f"mention_encodings: {len(mens)}")

        input_ids_and_labels = {
            "con_prop_encodings": con_prop_encodings,
            "property_encodings": property_encodings,
            "con_prop_labels": con_prop_labels,
            ###
            "con_def_encodings": con_def_encodings,
            "definition_encodings": definition_encodings,
            "con_defs_labels": con_defs_labels,
            ###
            "con_mens_encodings": con_mens_encodings,
            "mention_encodings": mention_encodings,
            "con_mens_labels": con_mens_labels,
        }

        return input_ids_and_labels


class JointConceptPropDefMen(nn.Module):
    def __init__(self, model_params):
        super(JointConceptPropDefMen, self).__init__()

        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params["hf_model_path"]

        _, model_class, _, self.mask_token_id = CLASSES[self.hf_checkpoint_name]

        log.info(f"model_class : {model_class}")

        self.concept_encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )
        self.property_encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )
        self.definition_encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )
        self.mention_encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )

        self.run_mode = model_params["run_mode"]
        self.loss_type = model_params["loss_type"]

        if self.run_mode == "train":
            self.cross_entropy_loss = nn.BCEWithLogitsLoss()

            self.miner = miners.MultiSimilarityMiner()
            self.use_hard_pair = model_params["use_hard_pair"]
            self.contrastive_loss_fn = losses.NTXentLoss(
                temperature=model_params["tau"]
            )

        log.info(f"run_mode: {self.run_mode}")
        log.info(f"loss_type: {self.loss_type}")

    def get_mask_token_embeddings(self, input_ids, last_layer_hidden_states):
        _, mask_token_index = (input_ids == torch.tensor(self.mask_token_id)).nonzero(
            as_tuple=True
        )
        mask_vectors = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(last_layer_hidden_states, mask_token_index)
            ]
        )
        return mask_vectors

    def forward(self, input_ids_and_labels):
        # input_ids_and_labels = {
        #     "con_prop_encodings": con_prop_encodings,
        #     "property_encodings": property_encodings,
        #     "con_prop_labels": con_prop_labels,
        #     "con_def_encodings": con_def_encodings,
        #     "definition_encodings": definition_encodings,
        #     "con_defs_labels": con_defs_labels,
        #     "con_mens_encodings": con_mens_encodings,
        #     "mention_encodings": mention_encodings,
        #     "con_mens_labels": con_mens_labels,
        # }

        if input_ids_and_labels["property_encodings"]:
            prop_output = self.property_encoder(
                **input_ids_and_labels["property_encodings"]
            )
            prop_hidden_states = prop_output.hidden_states[-1]
            prop_masks = self.get_mask_token_embeddings(
                input_ids_and_labels["property_encodings"]["input_ids"],
                prop_hidden_states,
            )  #####

            # Check for getting property embeddings in inference mode
            if self.run_mode == "train":
                con_prop_output = self.concept_encoder(
                    **input_ids_and_labels["con_prop_encodings"]
                )
                con_prop_hidden_states = con_prop_output.hidden_states[-1]
                con_prop_masks = self.get_mask_token_embeddings(
                    input_ids_and_labels["con_prop_encodings"]["input_ids"],
                    con_prop_hidden_states,
                )  #####

                loss_cross_con_prop, _, _ = calculate_inbatch_cross_entropy_loss(
                    concept_embeddings=con_prop_masks,
                    property_embeddings=prop_masks,
                    loss_fn=self.cross_entropy_loss,
                    device=device,
                )
            else:
                loss_cross_con_prop = 0.0
                prop_masks = None
        else:
            loss_cross_con_prop = 0.0
            con_prop_masks = None
            prop_masks = None

        print(f"loss_cross_con_prop: {loss_cross_con_prop}", flush=True)

        if input_ids_and_labels["definition_encodings"]:
            def_output = self.definition_encoder(
                **input_ids_and_labels["definition_encodings"]
            )
            def_hidden_states = def_output.hidden_states[-1]
            def_masks = self.get_mask_token_embeddings(
                input_ids_and_labels["definition_encodings"]["input_ids"],
                def_hidden_states,
            )  #####

            # Check for getting definition embeddings in inference mode
            if self.run_mode == "train":
                con_def_output = self.concept_encoder(
                    **input_ids_and_labels["con_def_encodings"]
                )
                con_def_hidden_states = con_def_output.hidden_states[-1]
                con_def_masks = self.get_mask_token_embeddings(
                    input_ids_and_labels["con_def_encodings"]["input_ids"],
                    con_def_hidden_states,
                )  #####

                if self.loss_type == "contrastive":
                    # Contrastive loss
                    con_def_all_embed = torch.cat([def_masks, con_def_masks], dim=0)

                    con_defs_labels = torch.tensor(
                        input_ids_and_labels["con_defs_labels"]
                    ).to(device=device)
                    con_defs_labels = torch.cat(
                        [con_defs_labels, con_defs_labels], dim=0
                    )

                    loss_contra_con_def = self.contrastive_loss_fn(
                        con_def_all_embed, con_defs_labels
                    )
                elif self.loss_type == "cross_entropy":
                    loss_contra_con_def, _, _ = calculate_inbatch_cross_entropy_loss(
                        concept_embeddings=con_def_masks,
                        property_embeddings=def_masks,
                        loss_fn=self.cross_entropy_loss,
                        device=device,
                    )
            else:
                loss_contra_con_def = 0.0
                con_def_masks = None

        else:
            loss_contra_con_def = 0.0
            con_def_masks = None
            def_masks = None

        print(f"loss_contra_con_def: {loss_contra_con_def}", flush=True)

        if input_ids_and_labels["mention_encodings"]:
            men_output = self.mention_encoder(
                **input_ids_and_labels["mention_encodings"]
            )
            men_hidden_states = men_output.hidden_states[-1]
            men_masks = self.get_mask_token_embeddings(
                input_ids_and_labels["mention_encodings"]["input_ids"],
                men_hidden_states,
            )  #####

            # Check for getting mention embeddings in inference mode
            if self.run_mode == "train":
                con_men_output = self.concept_encoder(
                    **input_ids_and_labels["con_mens_encodings"]
                )
                con_men_hidden_states = con_men_output.hidden_states[-1]
                con_men_masks = self.get_mask_token_embeddings(
                    input_ids_and_labels["con_mens_encodings"]["input_ids"],
                    con_men_hidden_states,
                )  #####

                if self.loss_type == "contrastive":
                    # Contrastive loss
                    con_men_all_embed = torch.cat([men_masks, con_men_masks], dim=0)

                    con_mens_labels = torch.tensor(
                        input_ids_and_labels["con_mens_labels"]
                    ).to(device=device)

                    print(f"con_men_all_embed.shape: {con_men_all_embed.shape}")
                    print(
                        f"before_con_mens_labels.shape: {con_mens_labels.shape}",
                        flush=True,
                    )

                    con_mens_labels = torch.cat(
                        [con_mens_labels, con_mens_labels], dim=0
                    )

                    print(
                        f"after_con_mens_labels.shape: {con_mens_labels.shape}",
                        flush=True,
                    )

                    loss_contra_con_men = self.contrastive_loss_fn(
                        con_men_all_embed, con_mens_labels
                    )
                elif self.loss_type == "cross_entropy":
                    loss_contra_con_men, _, _ = calculate_inbatch_cross_entropy_loss(
                        concept_embeddings=con_men_masks,
                        property_embeddings=men_masks,
                        loss_fn=self.cross_entropy_loss,
                        device=device,
                    )

            else:
                loss_contra_con_men = 0.0
                con_men_masks = None

        else:
            loss_contra_con_men = 0.0
            con_men_masks = None
            men_masks = None

        print(f"loss_contra_con_men: {loss_contra_con_men}", flush=True)

        loss = loss_cross_con_prop + loss_contra_con_def + loss_contra_con_men

        print(f"total_loss: {loss}", flush=True)

        return_dict = {
            "loss": loss,
            ###
            "con_prop_masks": con_prop_masks,
            "prop_masks": prop_masks,
            ###
            "con_def_masks": con_def_masks,
            "def_masks": def_masks,
            ###
            "con_men_masks": con_men_masks,
            "men_masks": men_masks,
        }

        return return_dict


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
        train_dataset = DatasetConceptPropDefMen(train_file, dataset_params)
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
            drop_last=False,
        )

        log.info(f"Train Data DF shape : {train_dataset.data_df.shape}")
    else:
        log.info(f"Train File is Empty.")
        train_dataloader = None

    if valid_file is not None:
        val_dataset = DatasetConceptPropDefMen(valid_file, dataset_params)
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

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    # Creating Model
    model = JointConceptPropDefMen(model_params=model_params)
    model.to(device=device)

    if load_pretrained:
        log.info(f"load_pretrained is : {load_pretrained}")
        log.info(f"Loading Pretrained Model Weights From : {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

        log.info(f"Loaded Pretrained Model")

    # if torch.cuda.is_available():
    #     n_gpu = torch.cuda.device_count()
    #     if n_gpu > 1:
    #         logging.info(f"using_multiple_GPUs: {n_gpu}")
    #         model = nn.DataParallel(model)
    #     model.to(device=device)

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

    for epoch in trange(max_epochs, desc="Epoch"):
        log.info("Epoch {:} of {:}".format(epoch, max_epochs))
        train_loss = 0.0
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc="train")):
            train_input_ids_and_labels = train_dataset.get_ids(batch)
            model.zero_grad()
            output_dict = model(input_ids_and_labels=train_input_ids_and_labels)

            running_train_loss = output_dict["loss"]

            if isinstance(model, nn.DataParallel):
                running_train_loss = running_train_loss.mean()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            running_train_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += running_train_loss.item()

            if (step + 1) % 100 == 0:
                log.info(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {running_train_loss.item():.4f}"
                )

            del train_input_ids_and_labels
            del running_train_loss
            gc.collect()

        log.info(f"train_step: {step}")
        train_loss /= step + 1

        del step
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        for step, batch in enumerate(tqdm(val_dataloader, desc="val")):
            val_input_ids_and_labels = val_dataset.get_ids(batch)
            with torch.no_grad():
                output_dict = model(input_ids_and_labels=val_input_ids_and_labels)
                running_val_loss = output_dict["loss"]

            if isinstance(model, nn.DataParallel):
                running_val_loss = running_val_loss.mean()
            val_loss += running_val_loss.item()

            del val_input_ids_and_labels
            del running_val_loss
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
            log.info("Early stopping. Model trained.")
            log.info(f"Model saved at: {model_file}")

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

    log.info(f"gpu_device: {device}")

    param_dict = prepare_data_and_models(config=config)

    train(config=config, param_dict=param_dict)
else:
    log = logging.getLogger(__name__)
