import gc
import logging
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.append("/scratch/c.scmag3/property_augmentation/model")

import warnings
from argparse import ArgumentParser
from glob import glob

import torch
import torch.nn as nn
from concept_property_model import ConceptPropertyModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
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
from je_utils import compute_scores, read_config, set_seed, clean_text
from multitask_con_prop_men_def import JointConceptPropDefMen

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


class DatasetConceptProperty(Dataset):
    def __init__(self, concept_property_df, dataset_params):
        self.data_df = concept_property_df[["concept", "property", "label"]]

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
        property = self.data_df["property"][idx]
        label = self.data_df["label"][idx]

        return {"concept": concept, "property": property, "label": label}

    def get_sent_ids(self, batch):
        cons = batch["concept"]
        property = batch["property"]
        label = batch["label"]

        cons_batch = [f"{con} means {self.mask_token}" for con in cons]
        property_batch = [f"{prop} means {self.mask_token}" for prop in property]

        print(flush=True)
        print(f"cons_batch: {cons_batch}", flush=True)
        print(f"property_batch: {property_batch}", flush=True)
        print(f"label: {label}", flush=True)

        con_encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=cons_batch,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        prop_encoded_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=property_batch,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # concept_input_id,
        # concept_attention_mask,
        # concept_token_type_id,
        # property_input_id,
        # property_attention_mask,
        # property_token_type_id,

        encoded_dict = {
            "concept_input_id": con_encoded_dict["input_ids"],
            "concept_attention_mask": con_encoded_dict["attention_mask"],
            "concept_token_type_id": con_encoded_dict["token_type_ids"],
            "property_input_id": prop_encoded_dict["input_ids"],
            "property_attention_mask": prop_encoded_dict["attention_mask"],
            "property_token_type_id": prop_encoded_dict["token_type_ids"],
            "label": label,
        }

        return encoded_dict


class BiEncoderConceptProperty(nn.Module):
    def __init__(self, config):
        super(BiEncoderConceptProperty, self).__init__()

        model_params = config["model_params"]
        training_params = config["training_params"]

        self.con_prop_bienc = ConceptPropertyModel(model_params=model_params)
        self.bce_loss_function = nn.BCEWithLogitsLoss()

        load_pretrained = training_params["load_pretrained"]
        freeze_concept_encoder = training_params["freeze_concept_encoder"]

        if load_pretrained:
            pretrained_bienc_model_path = training_params["pretrained_bienc_model_path"]
            self.con_prop_bienc.load_state_dict(
                torch.load(pretrained_bienc_model_path, map_location=device)
            )
            log.info(
                f"Loading Pretrained BiENcoder Model From : {pretrained_bienc_model_path}"
            )

        if freeze_concept_encoder:
            for name, parameter in self.con_prop_bienc.named_parameters():
                print(f"layer_name: {name}", flush=True)
                if "concept_encoder" in name:
                    print(
                        f"before_false_parameter.requires_grad: {parameter.requires_grad}"
                    )
                    parameter.requires_grad = False
                    print(
                        f"after_false_parameter.requires_grad: {parameter.requires_grad}"
                    )
            log.info(f"Freezing Concept Encoder of the Biencoder Model")

    def forward(self, ids_dict, pretrained_concept_embeddings=None):
        label = ids_dict.pop("label")

        concept_embedding, property_embedding, _ = self.con_prop_bienc(**ids_dict)

        # logits = (
        #     (pretrained_concept_embeddings * property_embedding)
        #     .sum(-1)
        #     .reshape(property_embedding.shape[0], 1)
        # )

        # For the baseline model where we do not update concept embedding
        logits = (
            (concept_embedding * property_embedding)
            .sum(-1)
            .reshape(property_embedding.shape[0], 1)
        )

        loss = None
        if label is not None:
            loss = self.bce_loss_function(
                logits, label.reshape_as(logits).float().to(device)
            )

        return (loss, logits, concept_embedding, property_embedding)


def prepare_data_and_models(config):
    ############
    training_params = config["training_params"]
    dataset_params = config["dataset_params"]
    ############

    log.info(f"{'*' * 50}")
    log.info(
        f"Preparing the model and dataset for the fold number: {training_params['fold_num']}"
    )
    log.info(f"{'*' * 50}")

    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    max_epochs = training_params["max_epochs"]
    batch_size = training_params["batch_size"]

    train_fold_df = training_params["train_fold_df"]
    test_fold_df = training_params["test_fold_df"]

    num_workers = 4
    if train_fold_df is not None:
        train_dataset = DatasetConceptProperty(train_fold_df, dataset_params)
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

    if test_fold_df is not None:
        test_dataset = DatasetConceptProperty(test_fold_df, dataset_params)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        log.info(f"Test Data DF shape : {test_dataset.data_df.shape}")
    else:
        log.info(f"Test File is Empty.")
        test_dataloader = None

    ########### Creating Model ###########
    model = BiEncoderConceptProperty(config=config)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info(f"Using Multiple GPUs: {n_gpu}")
            model = nn.DataParallel(model)
        model.to(device=device)

    log.info(f"model_class : {model.__class__.__name__}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if training_params["lr_schedule"] == "linear":
        warmup_ratio = training_params["warmup_ratio"]
        total_training_steps = len(train_dataloader) * training_params["max_epochs"]
        num_warmup_steps = math.ceil(total_training_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
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
        "test_dataset": test_dataset,
        "test_dataloader": test_dataloader,
    }


def train(config, param_dict):
    model = param_dict["model"]
    scheduler = param_dict["scheduler"]
    optimizer = param_dict["optimizer"]

    train_dataset = param_dict["train_dataset"]
    train_dataloader = param_dict["train_dataloader"]

    training_params = config["training_params"]
    max_epochs = training_params["max_epochs"]
    fold_num = training_params["fold_num"]

    model_name = "fold_" + str(fold_num) + training_params["model_name"]
    save_dir = training_params["save_dir"]
    model_save_file = os.path.join(save_dir, model_name)
    log.info(f"Fold {fold_num}, model_save_file: {model_save_file}")

    # For baseline model we do not need mention embeddings
    # pretrained_con_embeds_path = training_params["pretrained_con_embeds_path"]
    # with open(pretrained_con_embeds_path, "rb") as embed_pkl:
    #     pretrained_con_embeds_dict = pickle.load(embed_pkl)

    for epoch in trange(max_epochs, desc="Epoch"):
        log.info("Epoch {:} of {:}".format(epoch, max_epochs))
        train_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
            print(flush=True)
            print(
                f"batch['concept']: {len(batch['concept'])}, {batch['concept']}",
                flush=True,
            )
            print(
                f"batch['property']: {len(batch['property'])}, {batch['property']}",
                flush=True,
            )

            # pretrained_con_embeds = torch.tensor(
            #     [pretrained_con_embeds_dict[con] for con in batch["concept"]]
            # ).to(device)

            # print(f"pretrained_con_embeds.shape: {pretrained_con_embeds.shape}")

            ids_dict = train_dataset.get_sent_ids(batch)
            ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

            model.zero_grad()
            # For baseline model
            # loss, _, _, _ = model(
            #     ids_dict, pretrained_concept_embeddings=pretrained_con_embeds
            # )

            loss, _, _, _ = model(ids_dict, pretrained_concept_embeddings=None)

            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 100 == 0:
                log.info(
                    f"Epoch [{epoch}/{max_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

            del batch
            del ids_dict
            # del pretrained_con_embeds
            del loss
            gc.collect()

        log.info(f"train_step: {step + 1}")
        train_loss /= step + 1
        log.info(
            "Epoch: %d | train loss: %.4f ",
            epoch,
            train_loss,
        )

    ###################################
    #### Testing The Model on Fold ####
    ###################################

    log.info(f"Testing the model on Fold : {fold_num}")
    print(f"Testing the model on Fold: {fold_num}")

    test_dataset = param_dict["test_dataset"]
    test_dataloader = param_dict["test_dataloader"]
    fold_test_label, fold_test_pred = [], []

    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader, desc="Test Iteration")):
        print(flush=True)
        print(
            f"batch['concept']: {len(batch['concept'])}, {batch['concept']}",
            flush=True,
        )
        print(
            f"batch['property']: {len(batch['property'])}, {batch['property']}",
            flush=True,
        )

        # pretrained_con_embeds = torch.tensor(
        #     [pretrained_con_embeds_dict[con] for con in batch["concept"]]
        # ).to(device)

        # print(f"pretrained_con_embeds.shape: {pretrained_con_embeds.shape}")

        ids_dict = test_dataset.get_sent_ids(batch)
        ids_dict = {key: value.to(device) for key, value in ids_dict.items()}
        test_batch_label = ids_dict["label"].cpu().numpy().flatten()

        with torch.no_grad():
            # loss, test_batch_logits, _, _ = model(
            #     ids_dict, pretrained_concept_embeddings=pretrained_con_embeds
            # )

            loss, test_batch_logits, _, _ = model(
                ids_dict, pretrained_concept_embeddings=None
            )

        test_batch_preds = torch.round(torch.sigmoid(test_batch_logits))
        test_batch_preds = test_batch_preds.cpu().numpy().flatten()

        fold_test_label.append(test_batch_label)
        fold_test_pred.append(test_batch_preds)

        del batch
        del ids_dict
        # del pretrained_con_embeds
        del loss
        gc.collect()

    fold_test_label = np.concatenate(fold_test_label, axis=0)
    fold_test_pred = np.concatenate(fold_test_pred, axis=0)

    print(flush=True)
    print(f"fold_test_label: {fold_test_label.shape}, {fold_test_label}", flush=True)
    print(f"fold_test_pred: {fold_test_pred.shape}, {fold_test_pred}", flush=True)
    print(flush=True)

    log.info(f"Finished Training on fold: {fold_num}.")
    log.info(f"fold_test_label.shape: {fold_test_label.shape}")
    log.info(f"fold_test_pred.shape: {fold_test_pred.shape}")
    log.info(f"Test scores on Fold: {fold_num}")

    scores = compute_scores(fold_test_label, fold_test_pred)
    for key, value in scores.items():
        log.info(f"{key} : {value}")

    del model
    gc.collect()

    return fold_test_label, fold_test_pred


def model_evaluation_property_cross_validation(config):
    num_folds = config["training_params"]["num_folds"]
    train_test_data_dir = config["training_params"]["train_test_data_dir"]

    all_label, all_pred = [], []
    for fold_num in range(num_folds):
        train_pkl = glob(f"{train_test_data_dir}/{fold_num}_train*")[0]
        test_pkl = glob(f"{train_test_data_dir}/{fold_num}_test*")[0]

        log.info(f"train_file_path: {train_pkl}")
        log.info(f"test_file_path: {test_pkl}")

        with open(train_pkl, "rb") as train_file, open(test_pkl, "rb") as test_file:
            train_df = pickle.load(train_file)
            test_df = pickle.load(test_file)

        train_df["concept"] = train_df["concept"].apply(clean_text)
        train_df["property"] = train_df["property"].apply(clean_text)

        test_df["concept"] = test_df["concept"].apply(clean_text)
        test_df["property"] = test_df["property"].apply(clean_text)

        log.info(f"fold: {fold_num}")
        log.info(f"train_df: {train_df}")
        log.info(f"test_df: {test_df}")

        config["training_params"]["fold_num"] = fold_num
        config["training_params"]["train_fold_df"] = train_df
        config["training_params"]["test_fold_df"] = test_df

        param_dict = prepare_data_and_models(config=config)
        fold_test_label, fold_test_pred = train(config=config, param_dict=param_dict)

        all_label.append(fold_test_label)
        all_pred.append(fold_test_pred)

    all_label = np.concatenate(all_label, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    print(flush=True)
    print(f"all_label: {all_label.shape}, {all_label}", flush=True)
    print(f"all_pred: {all_pred.shape}, {all_pred}", flush=True)
    print(flush=True)

    log.info(f"Cross Validation Finished...")
    log.info(f"all_label.shape: {all_label.shape}")
    log.info(f"all_pred.shape: {all_pred.shape}")
    log.info(f"Test scores of all the folds")

    scores = compute_scores(all_label, all_pred)
    for key, value in scores.items():
        log.info(f"{key} : {value}")


def model_evaluation_concept_property_cross_validation(config):
    pass


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

    if config["training_params"]["do_cv"]:
        cv_type = config["training_params"]["cv_type"]

    if cv_type == "model_evaluation_property_split":
        config["training_params"]["num_folds"] = 5

        log.info(f'Parameter do_cv : {config["training_params"]["do_cv"]}')
        log.info(
            "Cross Validation for Model Evaluation - Data Splited on Property basis"
        )
        log.info(f"Parameter cv_type : {cv_type}")
        log.info(f'num_folds: {config["training_params"]["num_folds"]}')

        model_evaluation_property_cross_validation(config)

    elif cv_type == "model_evaluation_concept_property_split":
        log.info(f'Parameter do_cv : {config["training_params"].get("do_cv")}')
        log.info(
            "Cross Validation for Model Evaluation - Data Splited on both Concept and Property basis"
        )
        log.info(f"Parameter cv_type : {cv_type}")

        config["training_params"]["num_folds"] = 9
        model_evaluation_concept_property_cross_validation(config)
