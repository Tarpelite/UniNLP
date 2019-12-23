from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


import torch.nn as nn
from torch.optim import Adam
import copy

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import MTDNNModelv3 as MTDNNModel
from transformers import MTDNNModelTaskEmbeddingV2 as TaskEmbeddingModel
from transformers import AdapterMTDNNModel 
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer

# task_id list: {POS:0, NER:1, Chunking:2, SRL:3}
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModel, BertTokenizer),
    "task_embedding":(BertConfig, TaskEmbeddingModel, BertTokenizer)
}


class Decoder(object):

    def __init__(self, 
                special_tokens_count = 2,
                sep_token = "[SEP]",
                sequence_a_segment_id = 0,
                cls_token_segment_id = 1,
                cls_token="[CLS]",
                pad_token=0,
                pad_token_segment_id=0,
                model_type="bert"):
        
        self.MODEL_CLASSES = {
            "bert":(BertConfig, MTDNNModel, BertTokenizer),
            "task_embedding":(BertConfig, TaskEmbeddingModel, BertTokenizer)
        }
        self.special_tokens_count = special_tokens_count
        self.sep_token = sep_token
        self.sequence_a_segment_id = sequence_a_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_token_segment_id = pad_token_segment_id
        self.model_type = model_type

    def get_labels(self, pos_labels_fn, ner_labels_fn, chunking_labels_fn, srl_labels_fn):

        with open(pos_labels_fn, "r") as f:
            labels_pos = f.read().splitlines()
            if "X" not in labels_pos:
                labels_pos += ["X"]
            self.labels_pos = labels_pos 
        
        with open(ner_labels_fn, "r") as f:
            labels_ner = f.read().splitlines()
            if "O" not in labels_ner:
                labels_ner = ["O"] + labels_ner
            self.labels_ner = labels_ner

        with open(chunking_labels_fn, "r") as f:
            labels_chunking = f.read().splitlines()
            



    def setup_model(self, no_cuda=False):
        self.device  =  torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        
        config_class, model_clss, tokenizer_class = self.MODEL_CLASSES[self.model_type]

        

        

        

special_tokens_count = 2
sep_token = "[SEP]"
sequence_a_segment_id = 0
cls_token_segment_id = 1
cls_token = "[CLS]"
pad_token = 0
pad_token_segment_id = 0

def decode_pos(args, input_text, model, tokenizer):

    # load labels
    label_path = args.label_pos
    with open(path, "r") as f:
        labels = f.readlines().splitlines()
    if "X" not in labels:
        labels = labels + ["X"]
    
    label_map = {label: i for i, label in enumerate(labels)}
    

    tokens = tokenizer.tokenize(input_text)
    

    max_seq_length = args.max_seq_length
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
    
    valid_length = len(tokens)
    tokens += [sep_token]
    segment_ids = [sequence_a_segment_id]*len(tokens)

    tokens = [cls_token]
    max_len = len(tokens)
    segment_ids += [cls_token_segment_id]

    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)

    input_ids += ([pad_token] * padding_length)
    input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids += ([pad_token_segment_id]*padding_length)

    with torch.no_grad():
        inputs = {"input_ids": input_ids, 
                  "input_mask": input_mask,
                  "task_id": 0, 
                  "token_type_ids":segment_ids}

        outputs = model(**inputs)
    
    _, _, logits = outputs[:3]
    pres = logits.squeeze().detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)[1:valid_length + 1]
    tokens = tokens[1:valid_length + 1]

    pos_results = []
    r_list = []
    for tk, pred in zip(token_list, preds):
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
            pos_results[-1] = pos_results[-1] 
        else:
            r_list.append(tk)
            pos_results.append(pred)
    
    return pos_results, r_list

def decode_ner(args, input_text, model, tokenizer):


 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    
    parser.add_argument("--labels_pos", default=None, type=str)
    parser.add_argument("--labels_ner", default=None, type=str)
    parser.add_argument("--labels_chunking", default=None, type=str)
    parser.add_argument("--labels_srl", default=None, type=str)

    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)

    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()

    
    input_text = "I have a dog and he likes playing with me."
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    
    args.device = device

    pad_token_label_id = CrossEntropyLoss().ignore_index
    args.model_type = args.model_type.lower() 
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels_ner,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          output_hidden_states=True)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, 
                                        num_labels_pos=num_labels_pos, 
                                        num_labels_ner=num_labels_ner,
                                        num_labels_chunking=num_labels_chunking,
                                        num_labels_srl=num_labels_srl,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        init_last=args.init_last)
    
    model.to(args.device)

    logger.info("Setup the model parameters %s", args)

    res_pos = decode_pos(args, input_text, model, tokenizer)
    res_ner = decode_ner(args, input_text, model, tokenizer)
    res_chunking = decode_chunking(args, input_text, model, tokenizer)
    res_srl = decode_srl(args, input_text, model, tokenizer)



    

