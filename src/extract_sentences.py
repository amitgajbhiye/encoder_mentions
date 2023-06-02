import re


def find_whole_word(w):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


if __name__ == "__main__":
    concept_input_file = "data/cnetpchatgpt/con_vocab_cnetchatgpt.txt"

    ############################
    # wiki_text_file = "data/cnetpchatgpt/dummy_10000_wikipedia.txt"
    hawk_wiki_text_file = "/scratch/c.scmag3/en_wikipedia/en_wikipedia.txt"
    ############################

    out_file = "data/cnetpchatgpt/con_wiki_sents.tsv"

    print(f"input_concept_file : {concept_input_file}", flush=True)
    print(f"wiki_text_file : {hawk_wiki_text_file}", flush=True, end="\n")

    # concepts to reterive the sentence for
    cons = [line.rstrip() for line in open(concept_input_file)]

    con_sentences = {}
    max_sentence_length = 32
    num_extract_sents = 500

    print(f"num_input_concepts : {len(cons)}", flush=True)
    print(f"num_extract_sents : {len(num_extract_sents)}", flush=True)
    print(f"max_sentence_length : {len(max_sentence_length)}", flush=True, end="\n")

    with open(hawk_wiki_text_file, "r", encoding="utf-8") as wikifile, open(
        out_file, "w"
    ) as outfile:
        for line in wikifile:
            sent = line.strip()

            if len(sent.split(" ")) > max_sentence_length:
                continue

            for con in cons:
                if find_whole_word(con)(sent) is not None:
                    if con in con_sentences:
                        current_num_sents = len(con_sentences[con])

                        if current_num_sents < num_extract_sents:
                            con_sentences[con].append(sent)

                            outfile.write(f"{con}\t{sent}\n")
                            outfile.flush()
                            print(f"{current_num_sents}\t{con}\t{sent}", flush=True)
                        else:
                            continue
                    else:
                        con_sentences[con] = []

    # print("con_sentences", flush=True)
    # print(con_sentences, flush=True, end="\n")

    print(f"num_sents_for_concepts", flush=True, end="\n")
    for key, value in con_sentences.items():
        print(f"{key} : {len(value)}", end="\n", flush=True)
