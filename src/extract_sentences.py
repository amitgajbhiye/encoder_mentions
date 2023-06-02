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

    print(f"input_concept_file : {concept_input_file}", flush=True)
    print(f"wiki_text_file : {wiki_text_file}", flush=True)

    max_sentence_length = 32

    # concepts to reterive the sentence for
    cons = [line.rstrip() for line in open(concept_input_file)][0:100]

    print(f"num_input_concepts : {len(cons)}")

    con_sentences = {}
    num_extract_sents = 2
    with open(wiki_text_file, "r", encoding="utf-8") as f:
        for line in f:
            sent = line.strip()

            if len(sent.split(" ")) > max_sentence_length:
                continue

            for con in cons:
                if find_whole_word(con)(sent) is not None:
                    if con in con_sentences:
                        if len(con_sentences[con]) < num_extract_sents:
                            con_sentences[con].append(sent)
                    else:
                        con_sentences[con] = []

    print("con_sentences", flush=True)
    print(con_sentences, flush=True)
