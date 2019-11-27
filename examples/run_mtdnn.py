from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, MTDNNModel,
                                  BertForTokenClassification, BertTokenizer,
                                  RobertaConfig, RobertaForTokenClassification, RobertaTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils_pos import (read_UD_examples, convert_examples_to_features)

from utils_pos import convert_examples_to_features as convert_examples_to_pos_features
from utils_ner import  convert_examples_to_features as convert_examples_to_ner_features
from utils_ner import get_labels , read_examples_from_file

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModel, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, tokenizer, mini_batch=4):

    # load POS dataset
    logger.info("Creating POS features from dataset file at %s", args.pos_data_dir)
    train_file = os.path.join(args.pos_data_dir, "train.txt")
    eval_file = os.path.join(args.pos_eval_dir, "eval.txt")
    POS_examples_train = read_UD_examples(train_file, is_training=True)
    POS_examples_eval = read_UD_examples(eval_file, is_training=False)

    pos_train_features = convert_examples_to_pos_features(examples = POS_examples_train, tokenizer=tokenizer, max_seq_length=args.max_seq_length, is_training=True)
    pos_eval_feartures = convert_examples_to_pos_features(examples = POS_examples_eval, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    # load NER dataset
    logger.info("Creating NER features from dataset file at %s", args.ner_data_dir)
    NER_examples_train = read_examples_from_file(args.ner_data_dir, "train")
    NER_examples_eval = read_examples_from_file(args.ner_data_dir, "dev")
    ner_label_list = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    ner_train_features = convert_examples_to_ner_features(NER_examples_train, ner_label_list, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=CrossEntropyLoss().ignore_index)

    ner_eval_features = convert_examples_to_ner_features(NER_examples_eval, ner_label_list, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=CrossEntropyLoss().ignore_index)
    

    # pack the dataset t into mini-batch Dt:
    pos_train_dataset = []
    i = 0
    while i + mini_batch < len(pos_train_features):
        t_input_ids = []
        t_input_mask = []
        t_segment_ids = []
        t_label_ids = []
        for j in range(i, i+ mini_batch):
            t_input_ids.append(pos_train_features[j].input_ids)
            t_input_mask.append(pos_train_features[j].input_mask)
            t_segment_ids.append(pos_train_features[j].segment_ids)
            t_label_ids.append(pos_train_features[j].label_ids)
        
        pos_train_dataset.append([t_input_ids, t_input_mask, t_segment_ids, t_label_ids])
    
    t_input_ids = []
    t_input_mask = []
    t_segment_ids = []
    t_label_ids = []
    for j in range(i, len(pos_train_features)):
        t_input_ids.append(pos_train_features[j].input_ids)
        t_input_mask.append(pos_train_features[j].input_mask)
        t_segment_ids.append(pos_train_features[j].segment_ids)
        t_label_ids.append(pos_train_features[j].label_ids)
    
    pos_train_dataset.append([t_input_ids, t_input_mask, t_segment_ids, t_label_ids])

    ner_train_dataset = []
    i = 0
    while i + mini_batch < len(pos_train_features):
        t_input_ids = []
        t_input_mask = []
        t_segment_ids = []
        t_label_ids = []
        for j in range(i, i+ mini_batch):
            t_input_ids.append(pos_train_features[j].input_ids)
            t_input_mask.append(pos_train_features[j].input_mask)
            t_segment_ids.append(pos_train_features[j].segment_ids)
            t_label_ids.append(pos_train_features[j].label_ids)
        
    pos_train_dataset.append([t_input_ids, t_input_mask, t_segment_ids, t_label_ids])
    
    t_input_ids = []
    t_input_mask = []
    t_segment_ids = []
    t_label_ids = []
    for j in range(i, len(ner_train_features)):
        t_input_ids.append(ner_train_features[j].input_ids)
        t_input_mask.append(ner_train_features[j].input_mask)
        t_segment_ids.append(ner_train_features[j].segment_ids)
        t_label_ids.append(ner_train_features[j].label_ids)
    
    ner_train_dataset.append([t_input_ids, t_input_mask, t_segment_ids, t_label_ids])

    train_dataset_list = [pos_train_dataset, ner_train_dataset]
    train_task_list = ["POS", "NER"]

    # eval dataset
    all_input_ids = torch.tensor([f.input_ids for f in pos_eval_feartures], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in pos_eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in pos_eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in pos_eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    pos_eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    pass



            
            


def train(args, train_dataset, model, tokenizer):
    pass



