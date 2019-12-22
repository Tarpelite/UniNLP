# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Semantic role labelling fine-tuning: utilities to work with CoNLL-2003 task.
In this version, the sentence will be parsed into pair.

[CLS] Barack Obama went to Paris [SEP] went [SEP]

"""


from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from tqdm import *
from multiprocessing import Pool, cpu_count

num_cpus = cpu_count()

logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, verb):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.verb = verb


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        
        for line in f.readlines():
            words = []
            labels = []
            inputs = line.strip().strip("\n").split("|||")
            lefthand_input = inputs[0].strip().split()
            righthand_input = inputs[1].strip().split() if len(inputs) > 1 else ['O' for _ in lefthand_input]

            words = lefthand_input[1:]
            labels = righthand_input
            verb  = words[int(lefthand_input[0])]

            assert len(words) == len(labels) 

            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels,
                                         verb=verb
                                        ))
            
            guid_index += 1

    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    SRL_labels = ["A0", "A1", "A2", "A3", "A4", "A5","AA", "AM", "V", "O", 
                  "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS", "AM-EXT", "AM-LOC", "AM-TM",
                  "AM-MNR", "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-REC", "AM-TMP"]
    label_map = {label: i for i, label in enumerate(label_list)}
    # label_BIO_map = {label: i for i, label in enumerate(["B", "I", "O"])}
    # label_CRO_map = {label: i for i, label in enumerate(["C", "R", "O"])}
    # label_SRL_map = {label: i for i, label in enumerate(SRL_labels)}

    features = []
    cnt_counts = []
    def process(example):
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        verb_tokens = tokenizer.tokenize(example.verb)

        cnt_counts.append(len(tokens))
        special_tokens_count = 3
        if len(tokens) + len(verb_tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count - len(verb_tokens))]
            label_ids = label_ids[:(max_seq_length - special_tokens_count - len(verb_tokens))]

        tokens_a_len = len(tokens)
        tokens += [sep_token] + verb_tokens + [sep_token]
        label_ids += [pad_token_label_id] + [pad_token_label_id]*len(verb_tokens) + [pad_token_label_id]

        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * (tokens_a_len + 1) + [sequence_b_segment_id] * (len(verb_tokens) + 1)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids

           
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
           
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        in_f  = InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids)
        features.append(in_f)
        return in_f 
    with Pool(num_cpus) as p:
        results = list(tqdm(p.imap(process, examples), total=len(examples)))

        
    # features = []
    # cnt_counts = []
    # # last_tokens = []
    # # last_label_ids = []
    # for (ex_index, example) in enumerate(tqdm(examples)):
    #     if ex_index % 10000 == 0:
    #         logger.info("Writing example %d of %d", ex_index, len(examples))

    #     tokens = []
    #     label_ids = []
    #     for word, label in zip(example.words, example.labels):
    #         word_tokens = tokenizer.tokenize(word)  
    #         tokens.extend(word_tokens)
    #         # Use the real label id for the first token of the word, and padding ids for the remaining tokens
    #         label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        

    #     verb_tokens = tokenizer.tokenize(example.verb)
    #     # if ex_index == 0:
    #     #     last_tokens = tokens[-64:]
    #     #     last_label_ids = label_ids[-64:]
        
    #     # else:
    #     #     tokens = last_tokens + tokens
    #     #     label_ids = last_label_ids + label_ids
    #     #     last_tokens= tokens[-64:]
    #     #     last_label_ids = label_ids[-64:]

    #     # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    #     cnt_counts.append(len(tokens))
    #     special_tokens_count = 3 
    #     if len(tokens) + len(verb_tokens) > max_seq_length - special_tokens_count:
    #         tokens = tokens[:(max_seq_length - special_tokens_count - len(verb_tokens))]
    #         label_ids = label_ids[:(max_seq_length - special_tokens_count - len(verb_tokens))]

    #     # The convention in BERT is:
    #     # (a) For sequence pairs:
    #     #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #     #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    #     # (b) For single sequences:
    #     #  tokens:   [CLS] the dog is hairy . [SEP]
    #     #  type_ids:   0   0   0   0  0     0   0
    #     #
    #     # Where "type_ids" are used to indicate whether this is the first
    #     # sequence or the second sequence. The embedding vectors for `type=0` and
    #     # `type=1` were learned during pre-training and are added to the wordpiece
    #     # embedding vector (and position vector). This is not *strictly* necessary
    #     # since the [SEP] token unambiguously separates the sequences, but it makes
    #     # it easier for the model to learn the concept of sequences.
    #     #
    #     # For classification tasks, the first vector (corresponding to [CLS]) is
    #     # used as as the "sentence vector". Note that this only makes sense because
    #     # the entire model is fine-tuned.
    #     tokens_a_len = len(tokens)
    #     tokens += [sep_token] + verb_tokens + [sep_token]
    #     label_ids += [pad_token_label_id] + [pad_token_label_id]*len(verb_tokens) + [pad_token_label_id]
        
    #     if sep_token_extra:
    #         # roberta uses an extra separator b/w pairs of sentences
    #         tokens += [sep_token]
    #         label_ids += [pad_token_label_id]
    
    #     segment_ids = [sequence_a_segment_id] * (tokens_a_len + 1) + [sequence_b_segment_id] * (len(verb_tokens) + 1)

    #     if cls_token_at_end:
    #         tokens += [cls_token]
    #         label_ids += [pad_token_label_id]
    #         segment_ids += [cls_token_segment_id]
            
    #     else:
    #         tokens = [cls_token] + tokens
    #         label_ids = [pad_token_label_id] + label_ids
    #         segment_ids = [cls_token_segment_id] + segment_ids

    #     input_ids = tokenizer.convert_tokens_to_ids(tokens)

    #     # The mask has 1 for real tokens and 0 for padding tokens. Only real
    #     # tokens are attended to.
    #     input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    #     # Zero-pad up to the sequence length.
    #     padding_length = max_seq_length - len(input_ids)
    #     if pad_on_left:
    #         input_ids = ([pad_token] * padding_length) + input_ids
    #         input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
    #         segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    #         label_ids = ([pad_token_label_id] * padding_length) + label_ids

           
    #     else:
    #         input_ids += ([pad_token] * padding_length)
    #         input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
    #         segment_ids += ([pad_token_segment_id] * padding_length)
    #         label_ids += ([pad_token_label_id] * padding_length)
           
    #     assert len(input_ids) == max_seq_length
    #     assert len(input_mask) == max_seq_length
    #     assert len(segment_ids) == max_seq_length
    #     assert len(label_ids) == max_seq_length
        
    #     if ex_index < 5:
    #         logger.info("*** Example ***")
    #         logger.info("guid: %s", example.guid)
    #         logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
    #         logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    #         logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    #         logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    #         logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

    #     # print(type(label_ids))
    #     features.append(
    #             InputFeatures(input_ids=input_ids,
    #                           input_mask=input_mask,
    #                           segment_ids=segment_ids,
    #                           label_ids=label_ids))
    
    logger.info("*** Statistics ***")
    logger.info("*** max_len:{}  min_len:{} avg_len:{}***".format(max(cnt_counts), min(cnt_counts), sum(cnt_counts) / len(cnt_counts)))

    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            labels = [x for x in labels if len(x) > 0]
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
