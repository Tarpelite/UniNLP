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

from utils_ner import convert_examples_to_features as convert_examples_to_features_ner
from utils_ner import get_labels as get_labels_ner
from utils_ner import read_examples_from_file as read_examples_from_file_ner


from utils_pos import convert_examples_to_features as convert_examples_to_features_pos
from utils_pos import get_labels as get_labels_pos
from utils_pos import read_examples_from_file as read_examples_from_file_pos

from utils_chunking import convert_examples_to_features as convert_examples_to_features_chunking
from utils_chunking import get_labels as get_labels_chunking
from utils_chunking import read_examples_from_file as read_examples_from_file_chunking

from utils_srl import convert_examples_to_features as convert_examples_to_features_srl
from utils_srl import get_labels as get_labels_srl
from utils_srl import read_examples_from_file as read_examples_from_file_srl

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



def decode_pos(args, input_text, model, tokenizer):


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



    

