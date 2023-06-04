import re
import sys


def find_whole_word(w):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    ############################
    # wiki_text_file = "data/cnetpchatgpt/dummy_10000_wikipedia.txt"
    hawk_wiki_text_file = "/scratch/c.scmag3/en_wikipedia/en_wikipedia.txt"
    ############################

    concept_input_file = "data/cnetpchatgpt/con_vocab_cnetchatgpt.txt"

    print(f"input_concept_file : {concept_input_file}", flush=True)
    print(f"wiki_text_file : {hawk_wiki_text_file}", flush=True, end="\n")

    # concepts to reterive the sentence for
    print(f"start: {start}", flush=True)
    print(f"end: {end}", flush=True)

    cons = [line.rstrip() for line in open(concept_input_file)][start:end]

    con_sentences = {}
    max_sentence_length = 32
    num_extract_sents = 500

    out_file = f"data/cnetpchatgpt/con_{start}_{end}_wiki_sents.tsv"

    print(f"num_input_concepts : {len(cons)}", flush=True)
    print(f"num_extract_sents : {num_extract_sents}", flush=True)
    print(f"max_sentence_length : {max_sentence_length}", flush=True, end="\n")

    with open(hawk_wiki_text_file, "r", encoding="utf-8") as wikifile, open(
        out_file, "w"
    ) as outfile:
        wiki_line_count = 0
        for line in wikifile:
            print(f"wiki_line_count : {wiki_line_count}", flush=True)
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
                            print(
                                f"{current_num_sents}\t{con}\t{sent}",
                                flush=True,
                            )
                        else:
                            continue
                    else:
                        con_sentences[con] = []
            wiki_line_count += 1

    # print("con_sentences", flush=True)
    # print(con_sentences, flush=True, end="\n")

    print(f"num_sents_for_concepts", flush=True, end="\n")
    for key, value in con_sentences.items():
        print(f"{key} : {len(value)}", end="\n", flush=True)
