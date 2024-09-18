import torch
import os
import gc
import torch.nn as nn

import numpy as np
import pandas as pd
import logging
import time
import re
import sys

sys.path.insert(0, os.getcwd())

from get_mention_embeddings import ModelMentionEncoder as mention_encoder
from get_definition_embeddings import ModelDefinitionEncoder as definition_encoder


from je_utils import read_config, set_seed, compute_scores

from tqdm import tqdm, trange
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import BertModel, BertForMaskedLM, BertTokenizer
from torch.nn import BCEWithLogitsLoss


from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

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


CLASSES = {
    "bert-base-uncased": (BertModel, BertForMaskedLM, BertTokenizer, 103),
    "bert-large-uncased": (
        BertModel,
        BertForMaskedLM,
        BertTokenizer,
        103,
    ),
}


class WiCTSVDataset(Dataset):
    def __init__(self, datatype, dataset_params, test_df=None):
        if datatype == "train":
            self.file_path = dataset_params["train_file_path"]
        elif datatype == "valid":
            self.file_path = dataset_params["val_file_path"]
        elif datatype == "test":
            self.test_df = test_df

        self.data_df = pd.read_csv(self.file_path, sep="\t")

        log.info(f"datatype: {datatype}")
        log.info(f"file_path: {self.file_path}")
        log.info(f"dataframe_columns: {self.data_df.columns}")
        log.info(f"loaded_dataframe: {self.data_df}")

        if datatype == "test":
            assert "domain" in self.data_df, "Test data do not have domain columns"

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
        word = self.data_df["word"][idx]
        contexts = self.data_df["contexts"][idx]
        definitions = self.data_df["definitions"][idx]

        labels = self.data_df["labels"][idx]

        return {
            "word": word,
            "contexts": contexts,
            "definitions": definitions,
            "labels": labels,
        }

    def mask_word_in_context(self, search_string, input_string):
        def has_metacharacters(word):
            metacharacters = r"[]().*+?|{}^$\\"
            return [char for char in word if char in metacharacters]

        if has_metacharacters(word=search_string):
            raw_search_string = re.escape(search_string)
        else:
            # raw_search_string = r"\b" + search_string + r"\b"
            raw_search_string = search_string

        srch_output = re.search(raw_search_string, input_string, flags=re.IGNORECASE)

        start_idx = srch_output.start()
        end_idx = srch_output.end()
        mask_sent = input_string[:start_idx] + self.mask_token + input_string[end_idx:]

        return mask_sent

    def get_context_ids(self, batch):
        masked_contexts = [
            self.mask_word_in_context(word, context)
            for word, context in zip(batch["word"], batch["contexts"])
        ]

        # print(f"masked_contexts", flush=True)
        # print(masked_contexts, flush=True)

        encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=masked_contexts,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # encoded_dict["labels"] = batch["labels"]

        return encoded_dict

    def get_definition_ids(self, batch):
        masked_definitions = [
            f"{self.mask_token}: {definition}" for definition in batch["definitions"]
        ]

        # print(f"masked_definitions", flush=True)
        # print(masked_definitions, flush=True)

        encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=masked_definitions,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # encoded_dict["labels"] = batch["labels"]

        return encoded_dict


class SupervisedWicTsv(nn.Module):
    def __init__(self, model_params):
        super(SupervisedWicTsv, self).__init__()

        pretrained_mention_model_path = model_params["pretrained_mention_model_path"]
        pretrained_definition_model_path = model_params[
            "pretrained_definition_model_path"
        ]

        # Creating Mention Model
        self.men_model = nn.DataParallel(mention_encoder(model_params=model_params))
        self.men_model.to(device=device)

        # Creating Definition Model
        self.def_model = nn.DataParallel(definition_encoder(model_params=model_params))
        self.def_model.to(device=device)

        # Loading pretrained models

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
        # dropout_prop = model_params["dropout_prob"]
        # self.dropout = nn.Dropout(dropout_prop)
        # self.classifier = nn.Linear(2 * 1024, 1)

    def forward(self, context_ids_dict, definition_ids_dict, labels=None):
        mention_embeds = self.men_model(pretrained_con_embeds=None, **context_ids_dict)
        definition_embeds = self.def_model(
            pretrained_con_embeds=None, **definition_ids_dict
        )

        # concatenated_embeds = torch.cat((mention_embeds, definition_embeds), dim=1)

        print(f"mention_embeds.shape : {mention_embeds.shape}", flush=True)
        print(f"definition_embeds.shape : {definition_embeds.shape}", flush=True)

        # print(f"concatenated_embeds.shape : {concatenated_embeds.shape}", flush=True)

        # concatenated_embeds = nn.ReLU()(concatenated_embeds)
        # concatenated_embeds = self.dropout(concatenated_embeds)
        # logits = self.classifier(concatenated_embeds)

        logits = (
            (mention_embeds * definition_embeds)  # Elementwise product
            .sum(-1)
            .reshape(mention_embeds.shape[0], 1)
        )

        outputs = (logits,)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.reshape_as(logits).float().to(device))
            outputs = (loss,) + outputs

        return outputs

    ###################################################################


def prepare_data_and_models(config):
    ############
    training_params = config["training_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    # load_pretrained = training_params["load_pretrained"]

    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    max_epochs = training_params["max_epochs"]
    batch_size = training_params["batch_size"]

    train_file = dataset_params["train_file_path"]
    valid_file = dataset_params["val_file_path"]

    num_workers = 4

    if train_file is not None:
        train_dataset = WiCTSVDataset(datatype="train", dataset_params=dataset_params)
        train_sampler = RandomSampler(train_dataset)
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
        val_dataset = WiCTSVDataset(datatype="valid", dataset_params=dataset_params)
        val_sampler = RandomSampler(val_dataset)

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

    ########### Creating Model ###########
    model = SupervisedWicTsv(model_params=model_params)

    total_model_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"total_model_params: {total_model_params}", flush=True)
    print(f"total_trainable_params: {total_trainable_params}", flush=True)

    log.info(f"total_model_params: {total_model_params}")
    log.info(f"total_trainable_params: {total_trainable_params}")

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

    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in trange(max_epochs, desc="Epoch"):
        log.info(f"{'*' * 80}")
        log.info("Epoch {:} of {:}".format(epoch + 1, max_epochs))
        train_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            labels = batch["labels"]
            # print(f"batch_labels: {type(labels), labels}", flush=True)
            context_ids_dict = train_dataset.get_context_ids(batch)
            context_ids_dict = {
                key: value.to(device) for key, value in context_ids_dict.items()
            }

            definition_ids_dict = train_dataset.get_definition_ids(batch)
            definition_ids_dict = {
                key: value.to(device) for key, value in definition_ids_dict.items()
            }

            model.zero_grad()
            outputs = model(
                context_ids_dict=context_ids_dict,
                definition_ids_dict=definition_ids_dict,
                labels=labels,
            )
            loss, logits = outputs

            print(flush=True)
            print(f"train_batch_logits: {type(logits), logits}", flush=True)
            print(f"train_batch_label: {type(labels), labels}", flush=True)

            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 100 == 0:
                labels = labels.reshape(-1, 1).detach().cpu().numpy()
                logits = (
                    torch.round(torch.sigmoid(logits))
                    .reshape(-1, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                scores = compute_scores(labels, logits)

                log.info(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Accuracy: {scores['accuracy']}"
                )

            del context_ids_dict
            del definition_ids_dict
            del labels
            del logits
            del loss
            gc.collect()

        log.info(f"train_step: {step}")
        train_loss /= step + 1

        # Validation

        log.info(f"Running Validation ...")
        val_loss = 0.0
        all_labels, all_logits = [], []

        model.eval()
        for step, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
            labels = batch["labels"]
            print(f"val_batch_labels: {type(labels), labels}", flush=True)

            context_ids_dict = val_dataset.get_context_ids(batch)
            context_ids_dict = {
                key: value.to(device) for key, value in context_ids_dict.items()
            }

            definition_ids_dict = val_dataset.get_definition_ids(batch)
            definition_ids_dict = {
                key: value.to(device) for key, value in definition_ids_dict.items()
            }

            with torch.no_grad():
                outputs = model(
                    context_ids_dict=context_ids_dict,
                    definition_ids_dict=definition_ids_dict,
                    labels=labels,
                )

            loss, logits = outputs

            print("val_batch_logits: {logits}", flush=True)
            print("val_batch_lables: {labels}", flush=True)

            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            val_loss += loss.item()

            all_logits.extend(logits)
            all_labels.extend(labels)

            del context_ids_dict
            del definition_ids_dict
            del loss
            gc.collect()

        all_preds = (
            torch.round(torch.sigmoid(torch.vstack(all_logits)))
            .reshape(-1, 1)
            .detach()
            .cpu()
            .numpy()
        )
        all_labels = torch.vstack(all_labels).reshape(-1, 1).detach().cpu().numpy()

        print(f"val_all_logits: {len(all_logits)}, {all_logits}", flush=True)
        print(f"val_all_preds: {all_preds.shape}, {all_preds}", flush=True)
        print(f"val_all_lables: {all_labels.shape}, {all_labels}", flush=True)

        scores = compute_scores(all_labels, all_preds)
        log.info(f"validation_scores")
        for key, value in scores.items():
            log.info(f"{key}: {value}")

        running_val_accuracy = scores["accuracy"]

        if running_val_accuracy <= best_val_accuracy:
            patience_counter += 1

            log.info(f"Incrementing Patience Counter to: {patience_counter}")
            log.info(f"Previous Best Accuracy: {best_val_accuracy}")
            log.info(f"Current Accuracy: {running_val_accuracy}")

        else:
            model_save_file = os.path.join(
                save_dir, f"val_acc_{running_val_accuracy}_{model_name}.pt"
            )
            log.info(f"Saving Model to: {model_save_file}")

            log.info(f"Previous Best Accuracy: {best_val_accuracy}")
            log.info(f"Current Accuracy: {running_val_accuracy}")

            best_val_accuracy = running_val_accuracy
            patience_counter = 0

            torch.save(
                model.state_dict(),
                model_save_file,
            )

        if patience_counter >= patience_early_stopping:
            log.info(
                f"Early Stopping ---> Maximum Patience - {patience_counter} Reached !!"
            )
            break

    del model
    del step
    torch.cuda.empty_cache()
    gc.collect()

    return model_save_file


def test_best_model(config):
    log.info(f"{'*' * 30} Testing the Model {'*' * 30}")

    best_model_path = config["training_params"]["best_model_path"]

    log.info(f"best_model_path: {best_model_path}")

    dataset_params = config["dataset_params"]
    test_file = dataset_params["test_file_path"]
    test_df = pd.read_csv(test_file, sep="\t")

    training_params = config["training_params"]
    batch_size = training_params["batch_size"]

    log.info(f"test_file: {test_file}")
    log.info(f"test_df.columns: {test_df.columns}")
    log.info(f"test_df: {test_df}")

    model_params = config["model_params"]
    model = SupervisedWicTsv(model_params=model_params)
    model.load_state_dict(torch.load(best_model_path))

    test_domains = [
        ("0", "WNT_WKT"),
        ("1", "MSH"),
        ("2", "CTL"),
        ("3", "CPS"),
        ("4", "all"),
    ]

    for domain_id, domain in test_domains:
        log.info(f"******* Testing on domain_id: {domain_id}, domain: {domain} *******")

        if domain_id in ("0", "1", "2", "3"):
            domain_data_df = test_df[test_df["domain"] == domain_id]
        else:
            log.info(f"*** Testing on All Domains ***")
            domain_data_df = test_df

        log.info(f"num_test_instance : {len(domain_data_df)}")

        test_dataset = WiCTSVDataset(
            datatype="test", test_df=domain_data_df, dataset_params=dataset_params
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=None,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

        log.info(f"test_dataset.test_df: {len(test_dataset.test_df)}")
        log.info(f"test_dataset.test_df: {test_dataset.test_df.shape}")

        all_logits, all_labels = [], []

        model.eval()
        for step, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            labels = batch["labels"]

            # print(f"test_batch_labels: {type(labels), labels}", flush=True)
            # log.info(f"test_batch_labels: {type(labels), labels}")

            context_ids_dict = test_dataset.get_context_ids(batch)
            context_ids_dict = {
                key: value.to(device) for key, value in context_ids_dict.items()
            }

            definition_ids_dict = test_dataset.get_definition_ids(batch)
            definition_ids_dict = {
                key: value.to(device) for key, value in definition_ids_dict.items()
            }

            with torch.no_grad():
                outputs = model(
                    context_ids_dict=context_ids_dict,
                    definition_ids_dict=definition_ids_dict,
                    labels=labels,
                )

            loss, logits = outputs

            print("test_batch_logits: {logits}", flush=True)
            print("test_batch_lables: {labels}", flush=True)

            all_logits.extend(logits)
            all_labels.extend(labels)

            del context_ids_dict
            del definition_ids_dict
            del loss
            gc.collect()

        all_preds = (
            torch.round(torch.sigmoid(torch.vstack(all_logits)))
            .reshape(-1, 1)
            .detach()
            .cpu()
            .numpy()
        )
        all_labels = torch.vstack(all_labels).reshape(-1, 1).detach().cpu().numpy()
        print(flush=True)
        print(f"test_all_logits: {len(all_logits)}, {all_logits}", flush=True)
        print(f"test_all_preds: {all_preds.shape}, {all_preds}", flush=True)
        print(f"test_all_lables: {all_labels.shape}, {all_labels}", flush=True)

        scores = compute_scores(all_labels, all_preds)

        log.info(f"{'#' * 80}")

        log.info(f"test_scores on domain {domain_id}")
        for key, value in scores.items():
            log.info(f"{key}: {value}")


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

    lrs = [2e-6, 1e-5]
    batch_sizes = [4, 8]

    for lr in lrs:
        for batch_size in batch_sizes:
            model_name = f"lr_{lr}_bs_{batch_size}" + str(
                config["training_params"]["model_name"]
            )

            config["training_params"]["model_name"] = model_name
            config["training_params"]["lr"] = lr
            config["training_params"]["batch_size"] = batch_size

            log.info(f"new_configuration")
            log.info(f"model_name: {model_name}")

            param_dict = prepare_data_and_models(config=config)
            best_model_path = train(config=config, param_dict=param_dict)

            log.info(f"{'*' * 80}")
            log.info(f"training_finished")
            log.info(f"Best Model is Saved to {best_model_path}")

    # config["training_params"]["best_model_path"] = best_model_path
    # test_best_model(config)
