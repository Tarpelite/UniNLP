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
    return examples
    
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

def evaluate_pos(args, model):
    pos_examples = get_pos_examples(args.pos_data)
    print(len(pos_examples))

    pred_pos_labels = []
    true_labels = []
    total_words = []
    total_tokens = []
    # inferenece
    start = time.time()
    pos_label_list = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", 
        "DET", "INTJ", "NOUN", "NUM", "PART",
        "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
        "VERB", "X"
    ]

    # Just use the label of the first sub-word
    for exp in tqdm(pos_examples):

        words = exp[0]
        labels = exp[1]
        
        idxs = []
        text = ""
        for word in words:
            idxs += [len(text)]
            text += word + " "
        
        tokens = model(text)
        pred_pos_labels = []
        for tk in tokens:
            if tk.idx in idxs:
                pred_pos_labels.append(tk.pos_)

        # for word in words:
        #     tokens = model(word)
        #     total_words.append(word)
        #     total_tokens.append(tokens[0].text)
        #     pred_label = tokens[0].pos_
        #     if pred_label not in pos_label_list: 
        #         pred_label = "X"
        #         print(pred_label)
        #     pred_pos_labels.append(pred_label)
        
        assert len(pred_pos_labels) == len(labels)
       
    end = time.time()
    for exp in pos_examples:
        true_labels.extend(exp[1])

    res = []
    for l_pred in pred_pos_labels:
        res.extend(l_pred)
    pred_pos_labels = res
    ## evaluate
    total = len(pred_pos_labels)
    hit = 0
    for pred, true_label in zip(tqdm(pred_pos_labels), tqdm(true_labels)):
        if pred == true_label:
            hit += 1
    
    pre_start = time.time()
    

    test_data = [" ".join(exp[0]) for exp in pos_examples]
    for exp in tqdm(test_data):
        words = " ".join(exp)
        tokens = model(words)
    pre_end = time.time()

    print("sents per second", total*1.0000000/(pre_end - pre_start))
    print("pos tag time cost", end - start)
    print("pos acc", hit*1.0000000 / total)

    # write the prediction for checking
    with open("pred_pos.txt", "w+", encoding="utf-8") as f:
        for word, tok, pred, true in zip(total_words, total_tokens, pred_pos_labels, true_labels):
            line = word + "\t" + tok + "\t" + pred + "\t" + true
            f.write(line + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_data", type=str, default="")
    parser.add_argument("--ner_data", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")

    args = parser.parse_args()

    nlp = spacy.load(args.model_type)
    
    evaluate_pos(args, nlp)

    # ner_examples = get_ner_examples(args.ner_data)

    # print(len(ner_examples))

    