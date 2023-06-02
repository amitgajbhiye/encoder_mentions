import re


def find_whole_word(w):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def write_txt(output_file, contents):
    with open(output_file, "a+", encoding="utf-8") as f:
        f.writelines(contents)


if __name__ == "__main__":
    concept_input_file = "data/cnetpchatgpt/con_vocab_cnetchatgpt.txt"
    wiki_text_file = "data/cnetpchatgpt/dummy_10000_wikipedia.txt"

    hawk_wiki_text_file = "/scratch/c.scmag3/en_wikipedia/en_wikipedia.txt"

    out_file = "data/cnetpchatgpt/dummy_con_sents.tsv"

    print(f"input_concept_file : {concept_input_file}", flush=True)
    print(f"wiki_text_file : {wiki_text_file}", flush=True)

    max_sentence_length = 32

    # concepts to reterive the sentence for
    cons = [line.rstrip() for line in open(concept_input_file)][0:3]

    print(f"num_input_concepts : {len(cons)}, {cons}", flush=True, end="\n")

    con_sentences = {}
    num_extract_sents = 5

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
                            print(f"{current_num_sents}, {con}\t{sent}", flush=True)
                        else:
                            continue
                    else:
                        con_sentences[con] = []

    print("con_sentences", flush=True)
    print(con_sentences, flush=True, end="\n")

    for key, value in con_sentences.items():
        print(f"{key} : {value}", end="\n", flush=True)
