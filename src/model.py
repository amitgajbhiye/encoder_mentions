import os
import sys
import time
import logging
import pickle
import math
import gc

sys.path.insert(0, os.getcwd())

import warnings
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer

from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from transformers import (
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from transformers import (
    DebertaV2Model,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from je_utils import compute_scores, read_config, set_seed
from pytorch_metric_learning import losses, miners

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


CLASSES = {
    "bert-base-uncased": (BertModel, BertForSequenceClassification, BertTokenizer, 103),
    "bert-large-uncased": (
        BertModel,
        BertForSequenceClassification,
        BertTokenizer,
        103,
    ),
    "roberta-base": (
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaTokenizer,
        50264,
    ),
    "roberta-large": (
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaTokenizer,
        50264,
    ),
    "deberta-v3-large": (
        DebertaV2Model,
        DebertaV2ForSequenceClassification,
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
                    "labels": int,
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

            log.info(f"Loaded Dataframe Shape: {self.data_df.shape}")

        else:
            raise TypeError(f"Input file type is not correct !!! - {concept_sent_file}")

        self.data_df.reset_index(inplace=True, drop=True)

        log.info("Input DF")
        log.info(self.data_df.head(n=20))

        self.hf_tokenizer_name = dataset_params["hf_tokenizer_name"]
        self.hf_tokenizer_path = dataset_params["hf_tokenizer_path"]

        _, _, tokenizer_class, _ = CLASSES[self.hf_tokenizer_name]

        log.info(f"tokenizer_class : {tokenizer_class}")

        self.tokenizer = tokenizer_class.from_pretrained(self.hf_tokenizer_path)

        self.max_len = dataset_params["max_len"]

        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        self.print_freq = 0

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        concept = self.data_df["concept"][idx]
        sent = self.data_df["sent"][idx]

        return {"concept": concept, "sent": sent}


class ModelMentionEncoder(nn.Module):
    def __init__(self, model_params):
        super(ModelMentionEncoder, self).__init__()

        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params["hf_model_path"]

        model_class, _, _, self.mask_token_id = CLASSES[self.hf_checkpoint_name]

        log.info(f"model_class : {model_class}")

        self.encoder = model_class.from_pretrained(self.hf_model_path)

        classifier_dropout = self.encoder.config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        self.loss_fn = losses.NTXentLoss(temperature=config.tau)

    def get_bienc_con_embed(con_list):
        pass

    def forward(self, concept, sentence, labels=None):
        output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        hidden_states = output.last_hidden_state

        print(f"hidden_states : {hidden_states.shape}", flush=True)

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

        print(f"mask_vectors :{mask_vectors.shape}", flush=True)

        mask_vectors = self.dropout(mask_vectors)
        mask_logits = self.classifier(mask_vectors).view(-1)

        loss = None
        if labels is not None:
            labels = labels.view(-1).float()
            loss = loss_fct(mask_logits, labels)

        print("Step loss :", loss, flush=True)

        return (loss, mask_logits, mask_vectors)


def prepare_data_and_models(
    config,
    train_file,
    valid_file=None,
    test_file=None,
):
    training_params = config["training_params"]

    load_pretrained = training_params["load_pretrained"]
    pretrained_model_path = training_params["pretrained_model_path"]
    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    max_epochs = training_params["max_epochs"]
    batch_size = training_params["batch_size"]

    dataset_params = config["dataset_params"]
    model_params = config["model_params"]

    num_workers = 4

    if train_file is not None:
        train_data = DatasetConceptSentence(train_file, dataset_params)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
        )

        log.info(f"Train Data DF shape : {train_data.data_df.shape}")
    else:
        log.info(f"Train File is Empty.")
        train_dataloader = None

    if valid_file is not None:
        val_data = DatasetConceptSentence(valid_file, dataset_params)
        val_sampler = RandomSampler(val_data)
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        log.info(f"Valid Data DF shape : {val_data.data_df.shape}")
    else:
        log.info(f"Validation File is Empty.")
        val_dataloader = None

    if test_file is not None:
        test_data = DatasetConceptSentence(test_file, dataset_params)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
        )
        log.info(f"Test Data DF shape : {test_data.data_df.shape}")
    else:
        log.info("Test File is Empty.")
        test_dataloader = None

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    remove_classifier_layer = model_params["remove_classifier_layer"]

    log.info(f"remove_classifier_layer : {remove_classifier_layer}")

    # Creating Model
    model = ModelMentionEncoder(model_params=model_params)

    if load_pretrained:
        log.info(f"load_pretrained is {load_pretrained}")
        log.info(f"Loading Pretrained Model Weights From : {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

        log.info(f"Loaded Pretrained Model")

    model.to(device)

    log.info("Model")
    log.info(model)
    log.info(f"Model Class : {model.__class__.__name__}")

    if train_file is not None:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * max_epochs
        warmup_ratio = training_params["warmup_ratio"]
        num_warmup_steps = math.ceil(total_steps * warmup_ratio)

        log.info(f"num_warmup_steps : {num_warmup_steps}")
        log.info(f"total_steps : {total_steps}")

        # num_warmup_steps = 0

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        assert model.context_id == train_data.context_id, (
            f"Model context_id should be equal to dataset context_id;"
            f"model.context_id is {model.context_id} and dataset context_id is {train_data.context_id}"
        )

        return (
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        )
    else:
        return (model, test_dataloader)


def evaluate(model, dataloader):
    model.eval()

    val_losses, val_preds, val_labels = [], [], []

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].squeeze().to(device)
        token_type_ids = batch["token_type_ids"].squeeze().to(device)
        attention_mask = batch["attention_mask"].squeeze().to(device)

        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        if isinstance(model, ModelConceptPropertyJoint) or isinstance(
            model, ModelAnyNumberLabel
        ):
            loss, logits, mask_vectors = outputs
            batch_preds = torch.round(torch.sigmoid(logits))

        elif isinstance(model, ModelSeqClassificationConPropJoint):
            loss, logits = outputs

            batch_probs = logits.softmax(dim=1).squeeze(0)
            batch_preds = torch.argmax(batch_probs, dim=1).flatten()

        val_losses.append(loss.item())

        val_preds.extend(batch_preds.cpu().detach().numpy())
        val_labels.extend(labels.cpu().detach().numpy())

    avg_val_loss = round(np.mean(val_losses), 4)

    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()

    return avg_val_loss, val_preds, val_labels


def train(
    training_params,
    model,
    scheduler,
    optimizer,
    train_dataloader,
    val_dataloader=None,
    test_dataloader=None,
    fold=None,
):
    max_epochs = training_params["max_epochs"]
    model_name = training_params["model_name"]
    save_dir = training_params["save_dir"]

    patience_early_stopping = training_params["patience_early_stopping"]

    best_valid_f1 = 0.0
    patience_counter = 0
    start_epoch = 1
    epoch_train_losses, epoch_valid_losses = [], []
    (
        best_model_to_test,
        best_model_path,
    ) = (
        None,
        None,
    )

    for epoch in range(start_epoch, max_epochs + 1):
        log.info("Epoch {:} of {:}".format(epoch, max_epochs))

        step_train_losses = []

        model.train()
        for step, batch in enumerate(train_dataloader):
            concept = batch[0]
            sentences = batch[1]  #########################################

            model.zero_grad()

            input_ids = batch["input_ids"].squeeze().to(device)
            token_type_ids = batch["token_type_ids"].squeeze().to(device)
            attention_mask = batch["attention_mask"].squeeze().to(device)
            labels = batch["labels"].to(device)

            # print(flush=True)
            # print(f"In Step {step}", flush=True)
            # print(f"input_ids.shape : {input_ids.shape}", flush=True)
            # print(f"attention_mask.shape : {attention_mask.shape}", flush=True)
            # print(f"labels.shape : {labels.shape}", flush=True)
            # print(f"attention_mask : {attention_mask[0]}", flush=True)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if isinstance(model, ModelConceptPropertyJoint) or isinstance(
                model, ModelAnyNumberLabel
            ):
                loss, logits, mask_vectors = outputs

            elif isinstance(model, ModelSeqClassificationConPropJoint):
                loss, logits = outputs

            step_train_losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 and not step == 0:
                log.info(
                    "   Batch {} of Batch {} ---> Batch Loss {}".format(
                        step, len(train_dataloader), round(loss.item(), 4)
                    )
                )

        avg_train_loss = round(np.mean(step_train_losses), 4)
        log.info("Average Train Loss :", avg_train_loss)

        if (val_dataloader is not None) and (fold is None):
            log.info(f"Running Validation ....")
            log.info(f"Model Class : {model.__class__.__name__}")

            valid_loss, valid_preds, valid_gold_labels = evaluate(
                model=model, dataloader=val_dataloader
            )

            scores = compute_scores(valid_gold_labels, valid_preds)
            valid_binary_f1 = scores["binary_f1"]

            if best_valid_f1 >= valid_binary_f1:
                log.info(
                    f"Current Binary F1 : {valid_binary_f1} is worse than or equal to previous best : {best_valid_f1}"
                )
                patience_counter += 1
            else:
                patience_counter = 0

                log.info(f"{'*' * 50}")
                log.info(
                    f"Current Binary F1 : {valid_binary_f1} is better than previous best : {best_valid_f1}"
                )
                log.info("Epoch :", epoch)
                log.info("   Best Validation F1:", best_valid_f1)

                best_valid_f1 = valid_binary_f1

                if best_model_path is not None:
                    log.info(f"Deleting the previous best model : {best_model_path}")
                    os.remove(best_model_path)

                best_model_path = os.path.join(
                    save_dir, f"val_score_{str(best_valid_f1)}_{model_name}"
                )

                log.info(
                    f"Saving the current best Model at Epoch {epoch}: {best_model_path}"
                )
                torch.save(model.state_dict(), best_model_path)
                log.info(f"The best model is saved at : {best_model_path}")

                best_model_to_test = best_model_path

            epoch_train_losses.append(avg_train_loss)
            epoch_valid_losses.append(valid_loss)

            log.info(f"valid_preds shape: {valid_preds.shape}")
            log.info(f"val_gold_labels shape: {valid_gold_labels.shape}")

            log.info(f"Current Validation F1 Score Binary {valid_binary_f1}")
            log.info(f"Best Validation F1 Score Yet : {best_valid_f1}")

            log.info("Validation Scores")
            for key, value in scores.items():
                log.info(f" {key} :  {value}")

            if patience_counter >= patience_early_stopping:
                log.info(f"Train Losses :", epoch_train_losses)
                log.info(f"Validation Losses: ", epoch_valid_losses)

                log.info(
                    f"Early Stopping ---> Patience , {patience_early_stopping} Reached !!!"
                )
                log.info(f"The Best Validation Binary F1 : {best_valid_f1}")

                log.info(f"The Best Model is saved at : {best_model_path}")
                break

    # if (test_dataloader is not None) and (fold is not None):
    if test_dataloader is not None:
        if fold is not None:
            # For finetuning - Property Split and concept_property split.
            # Fold will not be None

            best_model_path = os.path.join(save_dir, f"{fold}_{model_name}")
            torch.save(model.state_dict(), best_model_path)

            log.info(f"Testing the Model on Fold : {fold}")
            log.info(f"Testing the model after epochs : {max_epochs}")
            log.info(f"The model for fold {fold} is saved at : {best_model_path}")

        else:
            # For pretraining for the datasets with a test set available;
            # For pretraining the fold will be None

            log.info(f"Fold is : {fold}")
            best_model_path = best_model_to_test

        model.load_state_dict(torch.load(best_model_path))

        log.info(f"Testing the Model loaded from : {best_model_path}")

        _, test_preds, test_gold_labels = evaluate(
            model=model, dataloader=test_dataloader
        )

        scores = compute_scores(test_gold_labels, test_preds)

        log.info(f"Fold: {fold}, test_gold_labels.shape : {test_gold_labels.shape}")
        log.info(f"Fold: {fold}, test_preds.shape : {test_preds.shape}")

        assert (
            test_gold_labels.shape == test_preds.shape
        ), "shape of fold's labels not equal to fold's preds"

        log.info(f"************ Test Scores Fold {fold} ************")
        for key, value in scores.items():
            log.info(f" {key} :  {value}")

        model.to(torch.device("cpu"))
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return test_preds, test_gold_labels


################ Finetuning Code Starts Here ################


def do_cv(config):
    training_params = config["training_params"]

    cv_type = training_params["cv_type"]
    data_dir = training_params["data_dir"]
    # save_prefix = training_params["save_prefix"]

    log.info(f"CV Type : {cv_type}")

    if cv_type == "concept_split":
        pass

    #     train_file_base_name = "train_mcrae"
    #     test_file_base_name = "test_mcrae"

    #     train_file_name = os.path.join(
    #         data_dir, f"{save_prefix}_{train_file_base_name}.tsv"
    #     )
    #     test_file_name = os.path.join(
    #         data_dir, f"{save_prefix}_{test_file_base_name}.tsv"
    #     )

    #     log.info(f"Train File Name : {train_file_name}")
    #     log.info(f"Test File Name : {test_file_name}")

    #     (
    #         model,
    #         scheduler,
    #         optimizer,
    #         train_dataloader,
    #         val_dataloader,
    #         test_dataloader,
    #     ) = prepare_data_and_models(
    #         config=config,
    #         train_file=train_file_name,
    #         valid_file=None,
    #         test_file=test_file_name,
    #     )

    #     test_preds, test_gold_labels = train(
    #         training_params,
    #         model,
    #         scheduler,
    #         optimizer,
    #         train_dataloader,
    #         val_dataloader,
    #         test_dataloader,
    #         fold=None,
    #     )

    elif cv_type in ("property_split", "concept_property_split"):
        if cv_type == "property_split":
            num_fold = 5
            train_file_base_name = "train_prop_split_con_prop.pkl"
            test_file_base_name = "test_prop_split_con_prop.pkl"

        elif cv_type == "concept_property_split":
            num_fold = 9
            train_file_base_name = "--------------"
            test_file_base_name = "--------------"

        else:
            raise NameError(
                "Specify cv_type from : 'concept_split', 'property_split', 'concept_property_split'"
            )

        log.info(f"Number of Folds : {num_fold}")
        log.info(f"Data Dir : {data_dir}")
        log.info(f"Train File Base Name : {train_file_base_name}")
        log.info(f"Test File Base Name : {test_file_base_name}")

        all_folds_test_preds, all_folds_test_labels = [], []
        for fold in range(num_fold):
            log.info("*" * 50)
            log.info(f"Training the model on Fold : {fold} of {num_fold}")
            log.info("*" * 50)

            train_file_name = os.path.join(data_dir, f"{fold}_{train_file_base_name}")
            test_file_name = os.path.join(data_dir, f"{fold}_{test_file_base_name}")

            with open(train_file_name, "rb") as train_pkl, open(
                test_file_name, "rb"
            ) as test_pkl:
                train_df = pickle.load(train_pkl)
                test_df = pickle.load(test_pkl)

            log.info(f"Train File Name : {train_file_name}")
            log.info(f"Test File Name : {test_file_name}")

            (
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = prepare_data_and_models(
                config=config,
                train_file=train_df,
                valid_file=None,
                test_file=test_df,
            )

            assert (
                val_dataloader is None
            ), "Validation data should be None for McRae dataset finetuning"

            fold_test_preds, fold_test_gold_labels = train(
                training_params,
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                fold=fold,
            )

            all_folds_test_preds.extend(fold_test_preds)
            all_folds_test_labels.extend(fold_test_gold_labels)

        all_folds_test_preds = np.array(all_folds_test_preds).flatten()
        all_folds_test_labels = np.array(all_folds_test_labels).flatten()

        log.info(f"Shape of All Folds Preds : {all_folds_test_preds.shape}")
        log.info(f"Shape of All Folds Labels : {all_folds_test_labels.shape}")

        assert (
            all_folds_test_preds.shape == all_folds_test_labels.shape
        ), "shape of all folds labels not equal to all folds preds"

        log.info("*" * 50)
        log.info(f"Calculating the scores for All Folds")
        print(f"Calculating the scores for All Folds", flush=True)

        all_folds_scores = compute_scores(all_folds_test_labels, all_folds_test_preds)

        for key, value in all_folds_scores.items():
            log.info(f"{key} : {value}")
            print(f"{key} : {value}", flush=True)

        return all_folds_scores


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser(description="Joint Encoder Concept Property Model")

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
    pprint(config, sort_dicts=False)

    training_params = config["training_params"]  # Training Parameters

    pretrain = training_params["pretrain"]
    finetune = training_params["finetune"]

    log.info(f"Pretrain : {pretrain}")
    log.info(f"Finetune : {finetune}")
    hp_tuning = training_params["hp_tuning"]

    if pretrain:
        train_file = training_params["train_file_path"]
        valid_file = training_params["val_file_path"]
        test_file = training_params["test_file_path"]

        log.info(f"Train File  : {train_file}")
        log.info(f"Valid File  : {valid_file}")
        log.info(f"Test File : {test_file}")

        if not hp_tuning:
            (
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = prepare_data_and_models(
                config=config,
                train_file=train_file,
                valid_file=valid_file,
                test_file=test_file,
            )

            # assert (
            #     test_dataloader is None
            # ), "Test dataloader should be None for pretraining."

            train(
                training_params=training_params,
                model=model,
                scheduler=scheduler,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                fold=None,
            )
        else:
            log.info("Pretraining Grid Search - Hyperparameter Tuning")

            max_epochs = [8, 10]
            batch_size = [32]
            warmup_ratio = [0.04, 0.06, 0.1, 0.15]
            weight_decay = [0.01, 0.1]

            log.info(f"max_epochs : {max_epochs}")
            log.info(f"batch_size : {batch_size}")
            log.info(f"warmup_ratio : {warmup_ratio}")
            log.info(f"weight_decay : {weight_decay}")

            model_name = config["model_params"]["hf_checkpoint_name"]

            for me in max_epochs:
                for bs in batch_size:
                    for wr in warmup_ratio:
                        for wd in weight_decay:
                            discription_str = f"me{me}_bs{bs}_wr{wr}_wd{wd}"

                            config["training_params"]["max_epochs"] = me
                            config["training_params"]["batch_size"] = bs
                            config["training_params"]["warmup_ratio"] = wr
                            config["training_params"]["weight_decay"] = wd
                            config["training_params"]["model_name"] = (
                                "cnetp_pretrain"
                                + model_name.replace("-", "_")
                                + "_"
                                + discription_str
                                + ".pt"
                            )

                            log.info("\n")
                            log.info("*" * 50)

                            log.info(f"discription_str : {discription_str}")

                            log.info(
                                f"New Run : max_epochs: {me}, batch_size: {bs}, warmup_ratio : {wr}, weight_decay : {wd}"
                            )
                            log.info(
                                f"Model Name: {config['training_params']['model_name']}"
                            )
                            log.info(f"new_config_file")
                            log.info(config)

                            (
                                model,
                                scheduler,
                                optimizer,
                                train_dataloader,
                                val_dataloader,
                                test_dataloader,
                            ) = prepare_data_and_models(
                                config=config,
                                train_file=train_file,
                                valid_file=valid_file,
                                test_file=test_file,
                            )

                            # assert (
                            #     test_dataloader is None
                            # ), "Test dataloader should be None for pretraining."

                            train(
                                training_params=training_params,
                                model=model,
                                scheduler=scheduler,
                                optimizer=optimizer,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                fold=None,
                            )

    elif finetune:
        if not hp_tuning:
            all_folds_scores = do_cv(config=config)

        else:
            log.info("Grid Search - Hyperparameter Tuning")

            epochs = [8, 10, 12]
            batch_size = [32, 16]
            learning_rate = [1e-5]
            warmup_ratio = [0.06, 0.1, 0.15]
            weight_decay = [0.01, 0.1]

            log.info(f"Max Epochs :  {epochs}")
            log.info(f"Batch Sizes : {batch_size}")
            log.info(f"Learning Rates : {learning_rate}")
            log.info(f"Warmup Ratio : {warmup_ratio}")
            log.info(f"Weight Decay : {weight_decay}")

            hp_combination_list, all_folds_scores_lists = [], []

            for ep in epochs:
                for bs in batch_size:
                    for lr in learning_rate:
                        for wr in warmup_ratio:
                            for wd in weight_decay:
                                log.info("*" * 60)
                                log.info(
                                    f"New Run : Epoch: {ep}, Batch Size: {bs}, LR: {lr}, warmup_ratio: {wr}, weight_decay: {wd}"
                                )
                                log.info("*" * 60)

                                config["training_params"]["max_epochs"] = ep
                                config["training_params"]["batch_size"] = bs
                                config["training_params"]["lr"] = lr
                                config["training_params"]["warmup_ratio"] = wr
                                config["training_params"]["weight_decay"] = wd

                                print(flush=True)
                                print(
                                    f"Running With Params Max Epochs, Batch Size, LR, WR:",
                                    config["training_params"]["max_epochs"],
                                    config["training_params"]["batch_size"],
                                    config["training_params"]["lr"],
                                    config["training_params"]["warmup_ratio"],
                                    config["training_params"]["weight_decay"],
                                )

                                print(f"Running with new config", flush=True)
                                pprint(config, sort_dicts=False)

                                all_folds_scores = do_cv(config=config)
                                print(
                                    f"Above Resulst are Running With Params Max Epochs, Batch Size, LR :",
                                    config["training_params"]["max_epochs"],
                                    config["training_params"]["batch_size"],
                                    config["training_params"]["lr"],
                                    config["training_params"]["warmup_ratio"],
                                    config["training_params"]["weight_decay"],
                                )
                                print("One Run Finished !!!", flush=True)
                                print(flush=True)

                                log.info("*" * 60)
                                log.info(
                                    f"Epoch: {ep}, Batch Size: {bs}, LR: {lr}, warmup_ratio: {wr}, weight_decay: {wd}"
                                )

                                for key, value in all_folds_scores.items():
                                    log.info(f"{key} : {value}")
                                log.info("*" * 60)

                                hp_combination_list.append(
                                    f"Epoch: {ep}, Batch Size: {bs}, LR: {lr}"
                                )
                                all_folds_scores_lists.append(all_folds_scores)
