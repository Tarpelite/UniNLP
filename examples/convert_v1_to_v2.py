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

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import MTDNNModelv4 as MTDNNModel
from transformers import MTDNNModelTaskEmbeddingV2 as TaskEmbeddingModel
from transformers import AdapterMTDNNModel 
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer

def convert_model(src_path, target_path):
    model = torch.load(src_path)
    model_to_save = model.module if hasattr(model, "module") else model 
    model_to_save.save_pretrained(target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", type=str)
    parser.add_argument("--tgt_path", type=str)

    args = parser.parse_args()

    convert_model(args.src_path, args.tgt_path)
