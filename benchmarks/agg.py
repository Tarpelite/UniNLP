import os
from tqdm import *

file = "/mnt/tianyu/data-processing/OntoNotes-5.0-NER/v4/english/dev.txt"

data = []
results = []
with open(file, encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        data.append(line)

print(data[:5])
text = ""
flag = None
for line in tqdm(data):
    l = line.strip()
    l = ' '.join(l.split())
    ls = l.split(" ")
    if len(ls) >= 11:
        word = ls[3]
        pos = ls[4]
        cons = ls[5]
        ori_ner = ls[10]
        ner = ori_ner
        # print(word, pos, cons, ner)
        if ori_ner == "*":
            if flag==None:
                ner = "O"
            else:
                ner = "I-" + flag
        elif ori_ner == "*)":
            ner = "I-" + flag
            flag = None
        elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner)>2:
            flag = ori_ner[1:-1]
            ner = "B-" + flag
        elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner)>2 and flag == None:
            ner = "B-" + ori_ner[1:-1]

        text += "\t".join([word, pos, cons, ner]) + '\n'
    else:
        text += '\n'

print(text[:100])
with open("dev.bio", "w+", encoding="utf-8") as f:
    f.write(text)