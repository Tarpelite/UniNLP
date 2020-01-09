import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, recall_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from utils_ner import convert_examples_to_features as convert_examples_to_features_ner
from utils_ner import read_examples_from_file as read_examples_from_file_ner


from utils_pos import convert_examples_to_features as convert_examples_to_features_pos
from utils_pos import read_examples_from_file as read_examples_from_file_pos

from utils_chunking import convert_examples_to_features as convert_examples_to_features_chunking
from utils_chunking import read_examples_from_file as read_examples_from_file_chunking

from utils_srl import convert_examples_to_features as convert_examples_to_features_srl
from utils_srl import read_examples_from_file as read_examples_from_file_srl

from utils_onto_pos import convert_examples_to_features as convert_examples_to_features_onto_pos
from utils_onto_pos import read_examples_from_file as read_examples_from_file_onto_pos

from utils_onto_ner import convert_examples_to_features as convert_examples_to_features_onto_ner
from utils_onto_ner import read_examples_from_file as read_examples_from_file_onto_ner

from utils_onto import get_labels 


import torch.nn as nn
from torch.optim import Adam
import copy
import requests
import os

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME,BertForTokenClassification, BertTokenizer
from transformers import MTDNNModelv4 
from transformers import MTDNNModelTaskEmbeddingV2 as TaskEmbeddingModel
from transformers import AdapterMTDNNModel 
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
from uninlp import MTDNNModel

def get_labels(labels_path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels

def convert_model(src_path, config_path, container_path, labels_path, target_path, task):
    model = torch.load(src_path)
    model_to_save = model.module if hasattr(model, "module") else model
    config = BertConfig.from_pretrained(config_path,
                                        num_labels=2,
                                        cache_dir=None,
                                        output_hidden_states=True)
    
    tgt_model = MTDNNModel.from_pretrained(container_path,
                                           from_tf=bool(".ckpt" in args.model_name_or_path),
                                           config=config,
                                           labels_list = [get_labels(labels_path)],
                                           do_task_embedding=False,
                                           do_alpha=False,
                                           do_adapter=False,
                                           num_adapter_layers=2
                                           )
    tgt_model.bert = model.bert
    tgt_model.dropout = model.dropout
    tgt_model.classifier_list[0] = getattr(model, "classifier_{}".format(task.lower()))

    if not os.path.exists(target_path):
        os.mkdir(target_path)
    tgt_model.save_pretrained(target_path)
    model_path = os.path.join(target_path, "pytorch_model.bin")
    cp_command = "mv {} {}".format(model_path, src_path)
    os.system(cp_command)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", type=str)
    parser.add_argument("--tgt_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--container_path", type=str)
    parser.add_argument("--labels_path", type=str)
    parser.add_argument("--task", type=str)

    args = parser.parse_args()

    convert_model(args.src_path, args.container_path, args.labels_path, args.target_path, task)
