import json
import logging
import os
import time

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_config(config_file):
    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            config_dict = json.load(json_file)
            return config_dict
    else:
        return config_file


def compute_scores(labels, preds):
    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

    precision, r, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="binary"
    )

    scores = {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores


def calculate_inbatch_cross_entropy_loss(
    concept_embeddings, property_embeddings, loss_fn, device
):
    logits_pos_concepts = (
        (concept_embeddings * property_embeddings)
        .sum(-1)
        .reshape(property_embeddings.shape[0], 1)
    )
    labels_pos_concepts = torch.ones_like(
        logits_pos_concepts, dtype=torch.float32, device=device
    )

    print(flush=True)
    print(f"++++++++++ in_calculate_inbatch_cross_entropy_loss ++++++++++", flush=True)
    print(f"concept_embeddings.shape: {concept_embeddings.shape}")
    print(f"property_embeddings.shape: {property_embeddings.shape}")
    print(
        f"logits_pos_concepts: {logits_pos_concepts.shape, logits_pos_concepts}",
        flush=True,
    )
    print(
        f"labels_pos_concepts: {labels_pos_concepts.shape, labels_pos_concepts}",
        flush=True,
    )
    print(flush=True)

    concept_embeddings = concept_embeddings.unsqueeze(1)
    property_embeddings = property_embeddings.unsqueeze(0)

    logits_neg_concepts = torch.sum(concept_embeddings * property_embeddings, dim=2)
    logits_neg_concepts.fill_diagonal_(0.0)
    logits_neg_concepts = logits_neg_concepts[logits_neg_concepts != 0.0]
    logits_neg_concepts = logits_neg_concepts.reshape(logits_neg_concepts.shape[0], 1)
    labels_neg_concepts = torch.zeros_like(
        logits_neg_concepts, dtype=torch.float32, device=device
    )

    print(flush=True)
    print(
        f"logits_neg_concepts: {logits_neg_concepts.shape, logits_neg_concepts}",
        flush=True,
    )
    print(
        f"labels_neg_concepts: {labels_neg_concepts.shape, labels_neg_concepts}",
        flush=True,
    )
    print(flush=True)

    all_logits = torch.cat((logits_pos_concepts, logits_neg_concepts), dim=0)
    all_labels = torch.cat((labels_pos_concepts, labels_neg_concepts), dim=0)

    print(f"all_logits: {all_logits.shape, all_logits}", flush=True)
    print(f"all_labels: {all_labels.shape, all_labels}", flush=True)

    all_loss = loss_fn(all_logits, all_labels)
    print(flush=True)
    print(f"all_cross_entropy_loss: {all_loss}", flush=True)
    print(flush=True)

    return all_loss, all_logits, all_labels


def old_calculate_inbatch_cross_entropy_loss(
    concept_embeddings, property_embeddings, loss_fn, device
):
    batch_logits, batch_labels = [], []

    logits_pos_concepts = (
        (concept_embeddings * property_embeddings)
        .sum(-1)
        .reshape(property_embeddings.shape[0], 1)
    )
    labels_pos_concepts = torch.ones_like(
        logits_pos_concepts, dtype=torch.float32, device=device
    )

    # labels_pos_concepts = torch.ones_like(
    #     logits_pos_concepts.shape[0], dtype=torch.float32, device=device
    # )

    print(flush=True)
    print(f"++++++++++ in_calculate_inbatch_cross_entropy_loss ++++++++++", flush=True)
    print(
        f"logits_pos_concepts: {logits_pos_concepts.shape, logits_pos_concepts}",
        flush=True,
    )
    print(
        f"labels_pos_concepts: {labels_pos_concepts.shape, labels_pos_concepts}",
        flush=True,
    )
    print(flush=True)

    ##############
    loss_pos_concept = loss_fn(logits_pos_concepts, labels_pos_concepts)
    ##############

    concept_embeddings = concept_embeddings.unsqueeze(1)
    property_embeddings = property_embeddings.unsqueeze(0)

    logits_neg_concepts = torch.sum(concept_embeddings * property_embeddings, dim=2)
    logits_neg_concepts.fill_diagonal_(0.0)
    logits_neg_concepts = logits_neg_concepts.flatten()

    logits_neg_concepts = logits_neg_concepts.reshape(logits_neg_concepts.shape[0], -1)

    labels_neg_concepts = torch.zeros_like(
        logits_neg_concepts, dtype=torch.float32, device=device
    )

    # labels_neg_concepts = torch.zeros(
    #     logits_neg_concepts.shape[0], dtype=torch.float32, device=device
    # )

    print(
        f"logits_neg_concepts: {logits_neg_concepts.shape, logits_neg_concepts}",
        flush=True,
    )
    print(
        f"labels_neg_concepts: {labels_neg_concepts.shape, labels_neg_concepts}",
        flush=True,
    )
    print(flush=True)

    ##############
    loss_neg_concept = loss_fn(logits_neg_concepts, labels_neg_concepts)
    ##############

    print(flush=True)
    print(f"loss_pos_concept: {loss_pos_concept}", flush=True)
    print(f"loss_neg_concept: {loss_neg_concept}", flush=True)

    ##############
    total_loss = loss_pos_concept + loss_neg_concept
    print(f"total_loss_entropy: {total_loss}", flush=True)
    print(flush=True)
    ##############

    batch_logits.append(logits_pos_concepts.flatten())
    batch_labels.append(labels_pos_concepts.flatten())

    batch_logits.extend(logits_neg_concepts.flatten())
    batch_labels.extend(labels_neg_concepts.flatten())

    ########################

    all_logits = torch.cat((logits_pos_concepts, logits_neg_concepts), dim=0)
    all_labels = torch.cat((labels_pos_concepts, labels_neg_concepts), dim=0)

    print(f"all_logits: {all_logits.shape, all_logits}", flush=True)
    print(f"all_labels: {all_labels.shape, all_labels}", flush=True)

    all_loss = loss_fn(all_logits, all_labels)
    print(flush=True)
    print(f"total_loss_entropy: {total_loss}", flush=True)
    print(f"all_cross_entropy_loss: {all_loss}", flush=True)
    print(flush=True)

    ########################

    return total_loss, batch_logits, batch_labels


def clean_text(text):
    text = text.strip().split("_")
    text = " ".join(text)

    return text
