import json
import os
import pickle
import sys
from argparse import ArgumentParser
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity as cosine
from transformers import BertTokenizer

import log

from mention_encoder import ModelMentionEncoder as mention_enc
from definition_encoder import ModelDefinitionEncoder as definition_enc

from multitask_con_prop_men_def import JointConceptPropDefMen

from je_utils import read_config, set_seed

sys.path.insert(0, os.getcwd())

logger = log.get_logger("root")
logger.propagate = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_models_and_tokenizer(config):
    ############
    inference_params = config["inference_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    pretrained_mention_model_path = inference_params["pretrained_mention_model_path"]
    pretrained_definition_model_path = inference_params[
        "pretrained_definition_model_path"
    ]

    # Creating Mention Model
    men_model = nn.DataParallel(mention_enc(model_params=model_params))
    men_model.to(device=device)

    # Creating Definition Model
    def_model = nn.DataParallel(definition_enc(model_params=model_params))
    def_model.to(device=device)

    men_model.load_state_dict(torch.load(pretrained_mention_model_path))
    def_model.load_state_dict(torch.load(pretrained_definition_model_path))
    tokenizer = BertTokenizer.from_pretrained(dataset_params["hf_tokenizer_path"])

    print(f"Mention Model is loaded from : {pretrained_mention_model_path}", flush=True)
    print(
        f"Definition Model is loaded from : {pretrained_definition_model_path}",
        flush=True,
    )

    return men_model, def_model, tokenizer


def get_mention_embeds(mention_enc, tokenizer, context_sents):
    max_len = 120

    ids_dict = tokenizer.batch_encode_plus(
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
        _, mention_vectors = mention_enc(pretrained_con_embeds=None, **ids_dict)

    return mention_vectors.cpu().numpy()


def get_definition_embeds(definition_enc, tokenizer, definition_sents):
    max_len = 120

    ids_dict = tokenizer.batch_encode_plus(
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
        _, def_vectors = definition_enc(pretrained_con_embeds=None, **ids_dict)

    return def_vectors.cpu().numpy()


def load_multitask_model_and_tokenizer(config):
    ############
    inference_params = config["inference_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    ############

    pretrained_multitask_model_path = inference_params[
        "pretrained_multitask_model_path"
    ]

    # Creating Multitask Model
    multitask_model = JointConceptPropDefMen(model_params=model_params)
    multitask_model.to(device=device)
    multitask_model.load_state_dict(torch.load(pretrained_multitask_model_path))

    tokenizer = BertTokenizer.from_pretrained(dataset_params["hf_tokenizer_path"])

    print(
        f"Multitask Model is loaded from : {pretrained_multitask_model_path}",
        flush=True,
    )
    print(
        f"multitask_model_name : {multitask_model.__class__.__name__ }",
        flush=True,
    )

    return multitask_model, tokenizer


def multitask_get_context_mention_embeds(
    multitask_model, context_sents, definition_sents, tokenizer
):
    max_len = 120

    mention_encodings = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=context_sents,
        max_length=max_len,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )

    definition_encodings = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=definition_sents,
        max_length=max_len,
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
    with torch.no_grad():
        outputs = multitask_model(input_ids_and_labels)

    mention_vectors = outputs["men_masks"].cpu().numpy()
    def_vectors = outputs["def_masks"].cpu().numpy()

    return mention_vectors, def_vectors


def get_print_result(sample_group: dict, sample_result: dict):
    candidates = sample_group["candidates"]
    info = sample_group["common_ancestor_info"]

    print_str = "\n===============================================================================================\n"
    print_str += f"Number of Candidates: {len(candidates)}\n\n"

    # print_str += f"\nCommon {info['relation']} synset: {info['ancestor_name']}\n{wn.synset(info['ancestor_name']).definition()}\n\n"
    # print_str += f"\nCommon {info['relation']} synset: {info['ancestor_name']}\n{info['ancestor_definition']}\n\n"
    print_str += f"\nCommon {info['relation']} synset: {info['ancestor_name']}\n\n"

    for candidate_no, candidate in enumerate(candidates):
        print_str += f"\n{candidate_no+1}) Synset: {candidate['synset_name']} ({candidate['words_in_contexts'][0]})\n"
        print_str += f"Definition: {candidate['definition']}\n"
        print_str += f"Context: {candidate['contexts'][0]}\n"

    print_str += "\n\n"
    print_str += f"Predicted alignment: {sample_result['predicted_alignment']}\n"
    print_str += f"Alignment Score: {sample_result['alignment_score']}\n"

    return print_str


def find_best_allignment(scores: np.array):
    # scores should be n x n  array where n is number of candidates in a synset group
    syn_count = scores.shape[0]
    all_permutations = list(permutations(range(syn_count)))

    best_score = -(10**10)
    for perm_no, permutation in enumerate(all_permutations):
        score = 0
        for no in range(syn_count):
            score += scores[no, permutation[no]]

        if score > best_score:
            best_score = score
            predicted_alignment = np.array(permutation)
            # print(f"\nPermutation No: {perm_no}\nBest Score: {best_score}\nPredicted alignment: {predicted_alignment}")

    aligment_score = sum(np.array(range(syn_count)) == predicted_alignment) / syn_count
    return predicted_alignment, aligment_score


def main(config):
    inference_params = config["inference_params"]
    data_folder = inference_params["data_folder"]
    sentence_model = inference_params["sentence_model"]
    save_dir = inference_params["save_dir"]
    results_dir = os.path.join(save_dir, sentence_model)
    os.makedirs(results_dir, exist_ok=False)

    model_type = inference_params["model_type"]

    if model_type == "multitask":
        multitask_model, tokenizer = load_multitask_model_and_tokenizer(config=config)
    else:
        mention_enc, def_enc, tokenizer = load_models_and_tokenizer(config=config)

    datasets = [
        "CoDA-clean-easy.json",
        "CoDA-clean-hard.json",
        "CoDA-noisy-easy.json",
        "CoDA-noisy-hard.json",
    ]
    word_types = ["n", "v"]

    for dataset in datasets:
        data_file = os.path.join(data_folder, dataset)
        with open(data_file, "r") as handle:
            CoDA = json.load(handle)

        for word_type in word_types:
            logger.info(
                f"Evaluating {sentence_model} on {dataset.split('.')[0]}_{word_type}"
            )

            print_file = f"on_{dataset.split('.')[0]}_{word_type}_some_results_target_removed.txt"
            save_file = (
                f"on_{dataset.split('.')[0]}_{word_type}_results_target_removed.pickle"
            )

            f_out = open(os.path.join(results_dir, print_file), "w", encoding="UTF-8")
            f_out.close()

            all_results = []
            sample_groups = CoDA[word_type]

            for group_no, sample_group in enumerate(sample_groups):
                candidates = sample_group["candidates"]

                contexts = []
                definitions = []
                for cand_no, candidate in enumerate(candidates):
                    print(flush=True)

                    print(f"unmasked_context: {candidate['contexts'][0]}", flush=True)

                    context = candidate["contexts"][0].replace(
                        candidate["words_in_contexts"][0], tokenizer.mask_token
                    )
                    definition = (
                        tokenizer.mask_token + ":" + " " + candidate["definition"]
                    )

                    contexts.append(context)
                    definitions.append(definition)

                    print(f"word: {candidate['words_in_contexts'][0]}")
                    print(f"masked_context: {context}", flush=True)
                    print(f"definition: {definition}", flush=True)

                if model_type == "multitask":
                    (
                        context_embs,
                        definition_embs,
                    ) = multitask_get_context_mention_embeds(
                        multitask_model=multitask_model,
                        context_sents=contexts,
                        definition_sents=definitions,
                        tokenizer=tokenizer,
                    )
                else:
                    context_embs = get_mention_embeds(
                        mention_enc=mention_enc,
                        tokenizer=tokenizer,
                        context_sents=contexts,
                    )

                    definition_embs = get_definition_embeds(
                        definition_enc=def_enc,
                        tokenizer=tokenizer,
                        definition_sents=definitions,
                    )

                print(flush=True)
                print(f"context_embs.shape : {context_embs.shape}", flush=True)
                print(f"definition_embs.shape : {definition_embs.shape}", flush=True)

                scores = cosine(context_embs, definition_embs)
                predicted_alignment, aligment_score = find_best_allignment(scores)

                print(flush=True)
                print(f"scores.shape : {scores.shape}", flush=True)
                print(
                    f"predicted_alignment: {predicted_alignment.shape}, {predicted_alignment}",
                    flush=True,
                )
                print(
                    f"aligment_score: {aligment_score.shape}, {aligment_score}",
                    flush=True,
                )

                sample_result = {}
                sample_result["predicted_alignment"] = predicted_alignment
                sample_result["alignment_score"] = aligment_score

                all_results.append(sample_result)

                if (group_no + 1) % 20 == 0:
                    logger.info(
                        f"{group_no+1}/{len(sample_groups)} synset groups processed"
                    )

                if (group_no + 1) % (len(sample_groups) // 20) == 0:
                    with open(
                        os.path.join(results_dir, print_file), "a", encoding="UTF-8"
                    ) as f_out:
                        f_out.write(get_print_result(sample_group, sample_result))

            with open(os.path.join(results_dir, save_file), "wb") as handle:
                print(f"total_sample_groups : {len(sample_groups)}")
                print(
                    f"average_alignment on: {save_file}: {len(all_results)}, {round(np.average([item['alignment_score'] for item in all_results]), 2)}"
                )
                pickle.dump(all_results, handle)


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser(description="CoDA21 Unsupervised Evaluation")

    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="path to the configuration file",
    )

    args = parser.parse_args()
    config = read_config(args.config_file)

    main(config=config)
