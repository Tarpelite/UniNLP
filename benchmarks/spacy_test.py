import spacy
import os
from tqdm import *
import argparse
import time

def get_pos_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
    examples= []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f.readlines():
            if line == "\n":
                if words:
                   examples.append([words, labels])
                   words = []
                   labels = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[1])
                labels.append(line[3])
    
    if words:
        examples.append([words, labels])
    
def get_ner_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append([words, labels])
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1 :
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            examples.append([words, labels])
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_data", type=str, default="")
    parser.add_argument("--ner_data", type=str, default="")

    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    pos_examples = get_pos_examples(args.pos_data)
    total = 0
    hit = 0
    pred_pos_labels = []
    true_labels = []
    ## inference
    start = time.time()

    for exp in pos_examples:
        words = nlp(exp[0])
        labels = exp[1]
        
        assert len(words) == len(labels)
        pred_pos_labels.extend([token.pos_ for token in words])

    end = time.time()

    for exp in pos_examples:
        true_labels.extend(exp[1])

    ## evaluate
    total = len(pred_pos_labels)
    for pred, true_label in zip(tqdm(pred_pos_labels), tqdm(true_labels)):
        if pred == true_label:
            hit += 1
    
    print("time cost", end - start)
    print("pos acc", hit*1.0000000 / total)