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
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
)


from je_utils import read_config, set_seed
from torch.utils.data import DataLoader, Dataset, SequentialSampler

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

        # self.data_df = self.data_df.sample(frac=1)
        # self.data_df.reset_index(inplace=True, drop=True)
        # self.unique_cons = self.data_df["concept"].unique()

        # self.data_df["labels"] = 0
        # self.data_df.set_index("concept", inplace=True, drop=False)

        # for lbl, con in enumerate(self.unique_cons, start=1):
        #     self.data_df.loc[con, "labels"] = lbl
        # self.data_df.reset_index(inplace=True, drop=True)

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
        # labels = self.data_df["labels"][idx]

        # return {"concept": concept, "sent": sent, "labels": labels}
        return {"concept": concept, "sent": sent}

    def mask_word_in_sent(self, con, sent):
        srch = re.search(con, sent, re.IGNORECASE)
        mask_sent = sent.replace(sent[srch.start() : srch.end()], self.mask_token, 1)

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

        # encoded_dict["labels"] = batch["labels"]

        return encoded_dict


class ModelMentionEncoder(nn.Module):
    def __init__(self, model_params):
        super(ModelMentionEncoder, self).__init__()

        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params["hf_model_path"]

        _, model_class, _, self.mask_token_id = CLASSES[self.hf_checkpoint_name]

        log.info(f"model_class : {model_class}")

        self.encoder = model_class.from_pretrained(
            self.hf_model_path, output_hidden_states=True
        )

        self.miner = miners.MultiSimilarityMiner()
        self.loss_fn = losses.NTXentLoss(temperature=model_params["tau"])

        self.use_hard_pair = model_params["use_hard_pair"]

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
        print(f"mask_vectors :{mask_vectors.shape}", flush=True)

        # emb_all = torch.cat([mask_vectors, pretrained_con_embeds], dim=0)
        # print(f"emb_all :{emb_all.shape}", flush=True)

        # if labels is None:
        #     labels = torch.arange(mask_vectors.size(0))
        # labels = torch.cat([labels, labels], dim=0)
        # print(f"labels :{labels.shape}: {labels}", flush=True, end="\n")

        # if self.use_hard_pair:
        #     hard_pairs = self.miner(emb_all, labels)
        #     loss = self.loss_fn(emb_all, labels, hard_pairs)
        # else:
        #     loss = self.loss_fn(emb_all, labels)

        # return loss, mask_vectors

        return mask_vectors


def prepare_data_and_models(config):
    ############
    training_params = config["training_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    load_pretrained = training_params["load_pretrained"]
    pretrained_model_path = training_params["pretrained_model_path"]

    # lr = training_params["lr"]
    # weight_decay = training_params["weight_decay"]
    # max_epochs = training_params["max_epochs"]
    batch_size = training_params["batch_size"]

    # train_file = dataset_params["train_file_path"]
    # valid_file = dataset_params["val_file_path"]
    test_file = dataset_params["test_file_path"]

    num_workers = 4

    # if train_file is not None:
    #     train_dataset = DatasetConceptSentence(train_file, dataset_params)
    #     train_sampler = MPerClassSampler(
    #         train_dataset.data_df["labels"],
    #         m=1,
    #         batch_size=batch_size,
    #         length_before_new_iter=len(train_dataset.data_df["labels"]),
    #     )

    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=batch_size,
    #         sampler=train_sampler,
    #         collate_fn=None,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #     )

    #     log.info(f"Train Data DF shape : {train_dataset.data_df.shape}")
    # else:
    #     log.info(f"Train File is Empty.")
    #     train_dataset = None
    #     train_dataloader = None

    # if valid_file is not None:
    #     val_dataset = DatasetConceptSentence(valid_file, dataset_params)
    #     val_sampler = MPerClassSampler(
    #         val_dataset.data_df["labels"],
    #         m=1,
    #         batch_size=batch_size,
    #         length_before_new_iter=len(val_dataset.data_df["labels"]),
    #     )
    #     val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=batch_size,
    #         sampler=val_sampler,
    #         collate_fn=None,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #     )
    #     log.info(f"Valid Data DF shape : {val_dataset.data_df.shape}")
    # else:
    #     log.info(f"Validation File is Empty.")
    #     val_dataset = None
    #     val_dataloader = None

    if test_file is not None:
        con_sent_dataset = DatasetConceptSentence(test_file, dataset_params)
        con_sent_sampler = SequentialSampler(con_sent_dataset)

        con_sent_dataloader = DataLoader(
            con_sent_dataset,
            batch_size=batch_size,
            sampler=con_sent_sampler,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        con_sent_dataset = None
        con_sent_dataloader = None

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    # Creating Model
    model = nn.DataParallel(ModelMentionEncoder(model_params=model_params))
    model.to(device=device)

    if load_pretrained:
        log.info(f"load_pretrained is : {load_pretrained}")
        log.info(f"Loading Pretrained Model Weights From : {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

        log.info(f"Loaded Pretrained Model")

    # if torch.cuda.is_available():
    #     n_gpu = torch.cuda.device_count()
    #     if n_gpu > 1:
    #         logging.info(f"using multiple GPUs: {n_gpu}")
    #         model = nn.DataParallel(model)
    #     model.to(device=device)

    log.info(f"model_class : {model.__class__.__name__}")

    # if train_file:
    #     optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #     if training_params["lr_schedule"] == "linear":
    #         scheduler = get_linear_schedule_with_warmup(
    #             optimizer,
    #             len(train_dataloader) * 2,
    #             int(len(train_dataloader) * max_epochs),
    #         )
    #     else:
    #         scheduler = get_cosine_schedule_with_warmup(
    #             optimizer,
    #             len(train_dataloader) * 2,
    #             int(len(train_dataloader) * max_epochs),
    #             num_cycles=1.5,
    #         )

    # return {
    #     "model": model,
    #     "scheduler": scheduler,
    #     "optimizer": scheduler,
    #     "train_dataset": train_dataset,
    #     "train_dataloader": train_dataloader,
    #     "val_dataset": val_dataset,
    #     "val_dataloader": val_dataloader,
    #     "con_sent_dataset": con_sent_dataset,
    #     "con_sent_dataloader": con_sent_dataloader,
    # }

    return {
        "model": model,
        "con_sent_dataset": con_sent_dataset,
        "con_sent_dataloader": con_sent_dataloader,
    }


def run_model(config, param_dict):
    model = param_dict["model"]

    con_sent_dataset = param_dict["con_sent_dataset"]
    con_sent_dataloader = param_dict["con_sent_dataloader"]

    training_params = config["training_params"]
    dataset_params = config["dataset_params"]

    con_sent_embed = []
    model.eval()
    for step, batch in enumerate(tqdm(con_sent_dataloader, desc="Iter")):
        print(f"concept : {batch['concept']}", flush=True)
        print(f"sents : {batch['sent']}", flush=True)

        ids_dict = con_sent_dataset.get_sent_ids(batch)
        ids_dict = {key: value.to(device) for key, value in ids_dict.items()}

        with torch.no_grad():
            outputs = model(pretrained_con_embeds=None, **ids_dict)

        mask_vectors = outputs
        mask_vectors = mask_vectors.cpu()

        print(f"*************************************")
        print(f"mask_vectors.shape : {mask_vectors}")
        print(f"*************************************")

        for con, sent, embed in zip(batch[0], batch[1], mask_vectors):
            con_sent_embed.append((con, sent, embed))

    save_dir = training_params["save_dir"]
    dataset_name = dataset_params["dataset_name"]

    out_file_name = os.path.join(save_dir, dataset_name)

    with open(out_file_name, "wb") as pkl_file:
        pickle.dump(con_sent_embed, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

    log.info(f"Embeddings are saved in : {out_file_name}")
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
