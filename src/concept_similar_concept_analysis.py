import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

setup1_mcrae_bienc_conc_embeds = "/home/amitgajbhiye/Downloads/embeddings_con_prop/mention_encoder/setup1_bienc_mcrae_concept_embeddings.pkl"
mention_embed_pkl = "/home/amitgajbhiye/Downloads/embeddings_con_prop/mention_encoder/cnetpchatgpt_dex_mention_embeds_mcrae_concepts.pkl"


def transform(vecs):
    maxnorm = max([np.linalg.norm(v) for v in vecs])

    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm**2 - np.linalg.norm(v) ** 2)))

    return new_vecs


def get_concept_similar_mention(concept_embed_pkl, mention_embed_pkl=None):
    with open(concept_embed_pkl, "rb") as con_pkl:
        con_bienc_embed = pickle.load(con_pkl)

    print(type(con_bienc_embed), f"Num concepts: {len(con_bienc_embed)}")

    return con_bienc_embed


def read_concept_all_mention_embeds_pkl(file_path):
    # Reading concept and the list of mention embeddings into dictionary - con_embed_list_dict

    print(f"In read_concept_all_mention_embeds_pkl_function")
    with open(file_path, "rb") as pkl:
        con_sent_embed_list = pickle.load(pkl)

    con_mention_embeds_dict = {}
    con_mentions_dict = {}

    for con, sent, emb in con_sent_embed_list:
        if con not in con_mention_embeds_dict:
            con_mention_embeds_dict[con] = [emb]
            con_mentions_dict[con] = [sent]
        else:
            con_mention_embeds_dict[con].append(emb)
            con_mentions_dict[con].append(sent)

    print(f"total_concepts_in_dict: {len(con_mention_embeds_dict)}")
    print(f"sent_dict_total_concepts_in_dict: {len(con_mentions_dict)}")

    # print (f"Concepts and Number of Embeddings")
    # for key, value in con_mention_embeds_dict.items():
    #     print(f"{key}: num_embed:{len(value)}, num_mentions: {len(con_mentions_dict[key])}")

    # print (f"Concepts and Number of Sents")
    # for key, value in con_mentions_dict.items():
    #     # if len(value) < 50:
    #     print(f"{key}: {len(value)}")

    return con_mention_embeds_dict, con_mentions_dict


# con_bienc_embed = get_concept_similar_mention(concept_embed_pkl=setup1_mcrae_bienc_conc_embeds)
# con_mention_embeds_dict, con_mentions_dict = read_concept_all_mention_embeds_pkl(file_path=mention_embed_pkl)


# def get_con_nearest_mentions(
#     con_bienc_embed, con_mention_embeds_dict, con_mentions_dict
# ):
#     num_nearest_neighbours = 50

#     uniq_cons = sorted(con_bienc_embed.keys())
#     # print (uniq_cons)

#     with open("mcrae_concept_similar_mentions.tsv", "w") as con_men_sent_file:
#         con_men_sent_file.write(f"CONCEPT\tMENTION_SENTENCE\tDISTANCE\n")

#         for con_number, con in enumerate(uniq_cons):
#             print(f"Processing Concept Number {con_number} - {con}")

#             con_bi_embed = con_bienc_embed[con]
#             men_sents = con_mentions_dict[con]
#             mention_embeds = con_mention_embeds_dict[con]

#             print(f"concept: {con}")
#             # print (type(con_bi_embed))
#             print("Number of mentions", len(men_sents))
#             print("Number of embeddings", len(mention_embeds))
#             # print (type(men_sents), len(men_sents))
#             # print (type(mention_embeds[0]))
#             print()

#             zero_bienc_con_embeds = np.expand_dims(
#                 np.insert(con_bi_embed, 0, float(0)), 0
#             )
#             transformed_concept_mention_embeddings = np.asarray(
#                 transform(mention_embeds)
#             )

#             # print (f"zero_bienc_con_embeds: {zero_bienc_con_embeds.shape}")
#             print(
#                 f"transformed_concept_mention_embeddings: {transformed_concept_mention_embeddings.shape}"
#             )

#             if transformed_concept_mention_embeddings.shape[0] < num_nearest_neighbours:
#                 num_nearest_neighbours = transformed_concept_mention_embeddings.shape[0]
#                 print(f"*** num_nearest_neighbours: {num_nearest_neighbours}")

#             bienc_con_similar_mention_embeds = NearestNeighbors(
#                 n_neighbors=num_nearest_neighbours, radius=100.0, algorithm="brute"
#             ).fit(transformed_concept_mention_embeddings)

#             con_distances, con_indices = bienc_con_similar_mention_embeds.kneighbors(
#                 zero_bienc_con_embeds
#             )

#             print(f"con_indices: {con_indices}")
#             print(f"con_distances: {con_distances}")

#             for idx, distance in zip(
#                 np.squeeze(con_indices), np.squeeze(con_distances)
#             ):
#                 # print (f"{con}\t{men_sents[idx]}\t{distance}")

#                 con_men_sent_file.write(f"{con}\t{men_sents[idx]}\t{distance}\n")


# con_bienc_embed = con_bienc_embed
# con_mention_embeds_dict=con_mention_embeds_dict
# con_mentions_dict= con_mentions_dict

# get_con_nearest_mentions(con_bienc_embed, con_mention_embeds_dict, con_mentions_dict)

# con_mentions_dict["eel"]


eco_all_mentions = "/home/amitgajbhiye/Downloads/embeddings_con_prop/mention_encoder/epoch2_cnetpchatgpt_dex_mention_encoder_embeds/setup1_cnetpchatgpt_3dex_mention_encoder_economy_mention_embeddings.pkl"

(
    eco_con_mention_embeds_dict,
    eco_con_mentions_dict,
) = read_concept_all_mention_embeds_pkl(file_path=eco_all_mentions)
