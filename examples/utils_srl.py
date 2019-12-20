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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from tqdm import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, verb_seq, labels_BIO, labels_CRO, labels_SRL):
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
        self.labels_BIO = labels_BIO
        self.labels_CRO = labels_CRO
        self.labels_SRL = labels_SRL
        self.verb_seq = verb_seq # one-hot sequence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, verb_seq_ids, input_mask, segment_ids, label_ids, label_BIO_ids, label_CRO_ids, label_SRL_ids):
        self.input_ids = input_ids
        self.verb_seq_ids = verb_seq_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_BIO_ids = label_BIO_ids
        self.label_CRO_ids = label_CRO_ids
        self.label_SRL_ids = label_SRL_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        
        for line in f.readlines():
            words = []
            labels = []
            labels_BIO = [] # ["B", "I", "O"]
            labels_CRO = [] # ["R", "C", "O"]
            labels_SRL = [] # ["A0", "A1", ...]
            verb_seq = []
            inputs = line.strip().strip("\n").split("|||")
            lefthand_input = inputs[0].strip().split()
            righthand_input = inputs[1].strip().split() if len(inputs) > 1 else ['O' for _ in lefthand_input]

            words = lefthand_input[1:]
            labels = righthand_input
            verb_seq = [0]*(len(words))
            verb_seq[int(lefthand_input[0])] = 1
            for x in labels:
                if "B-" in x or "I-" in x:
                    BIO_label = x[0]
                    x = x[2:]
                else:
                    BIO_label = "O"
                if "C-" in x or "R-" in x:
                    CRO_label = x[0]
                    x = x[2:]
                else:
                    CRO_label = "O"
                
                if len(x) > 0 and x != "O":
                    SRL_label = x
                else:
                    SRL_label = "O"
               
                labels_BIO.append(BIO_label)
                labels_CRO.append(CRO_label)
                labels_SRL.append(SRL_label)

            assert len(words) == len(labels) == len(verb_seq) == len(labels_BIO) == len(labels_CRO) == len(labels_SRL) 

            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels,
                                         labels_BIO=labels_BIO,
                                         labels_CRO=labels_CRO,
                                         labels_SRL=labels_SRL, 
                                         verb_seq=verb_seq))
            
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
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    SRL_labels = ["A0", "A1", "A2", "A3", "A4", "A5","AA", "AM", "V", "O", 
                  "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS", "AM-EXT", "AM-LOC", 
                  "AM-MNR", "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-REC", "AM-TMP"]
    label_map = {label: i for i, label in enumerate(label_list)}
    label_BIO_map = {label: i for i, label in enumerate(["B", "I", "O"])}
    label_CRO_map = {label: i for i, label in enumerate(["C", "R", "O"])}
    label_SRL_map = {label: i for i, label in enumerate(SRL_labels)}

    features = []
    cnt_counts = []
    # last_tokens = []
    # last_label_ids = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        label_BIO_ids = []
        label_CRO_ids = []
        label_SRL_ids = []

        verb_seq_ids = []
        for word, IsVerb, label, label_BIO, label_CRO, label_SRL in zip(example.words, example.verb_seq, example.labels, example.labels_BIO, example.labels_CRO, example.labels_SRL):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            verb_seq_ids.extend([IsVerb] + [IsVerb]*(len(word_tokens) - 1))
            BIO_id = label_BIO_map[label_BIO]
            label_BIO_ids.extend([BIO_id] + [BIO_id]*(len(word_tokens) - 1))
            CRO_id = label_CRO_map[label_CRO]
            label_CRO_ids.extend([CRO_id] + [CRO_id]*(len(word_tokens) - 1))
            SRL_id = label_SRL_map[label_SRL]
            label_SRL_ids.extend([SRL_id] + [SRL_id]*(len(word_tokens) - 1))

        # if ex_index == 0:
        #     last_tokens = tokens[-64:]
        #     last_label_ids = label_ids[-64:]
        
        # else:
        #     tokens = last_tokens + tokens
        #     label_ids = last_label_ids + label_ids
        #     last_tokens= tokens[-64:]
        #     last_label_ids = label_ids[-64:]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        cnt_counts.append(len(tokens))
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            verb_seq_ids = verb_seq_ids[:(max_seq_length - special_tokens_count)]
            label_BIO_ids = label_BIO_ids[:(max_seq_length - special_tokens_count)]
            label_CRO_ids = label_CRO_ids[:(max_seq_length - special_tokens_count)]
            label_SRL_ids = label_SRL_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        verb_seq_ids += [0]
        label_BIO_ids += [pad_token_label_id]
        label_CRO_ids += [pad_token_label_id]
        label_SRL_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            verb_seq_ids += [0]
            label_BIO_ids += [pad_token_label_id]
            label_CRO_ids += [pad_token_label_id]
            label_SRL_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            verb_seq_ids += [0]
            label_BIO_ids += [pad_token_label_id]
            label_CRO_ids += [pad_token_label_id]
            label_SRL_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            label_BIO_ids = [pad_token_label_id] + label_BIO_ids
            label_CRO_ids = [pad_token_label_id] + label_CRO_ids
            label_SRL_ids = [pad_token_label_id] + label_SRL_ids
            verb_seq_ids = [0] + verb_seq_ids
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
            label_BIO_ids = ([pad_token_label_id] * padding_length) + label_BIO_ids
            label_CRO_ids = ([pad_token_label_id] * padding_length) + label_CRO_ids
            label_SRL_ids = ([pad_token_label_id] * padding_length) + label_SRL_ids
            verb_seq_ids = ([0] * padding_length) + verb_seq_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
            label_BIO_ids += ([pad_token_label_id] * padding_length)
            label_CRO_ids += ([pad_token_label_id] * padding_length)
            label_SRL_ids += ([pad_token_label_id] * padding_length)
            verb_seq_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(verb_seq_ids) == max_seq_length
        assert len(label_BIO_ids) == max_seq_length
        assert len(label_CRO_ids) == max_seq_length
        assert len(label_SRL_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("verb_seq_ids: %s", " ".join([str(x) for x in verb_seq_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              verb_seq_ids=verb_seq_ids,
                              segment_ids=segment_ids,
                              label_ids=label_ids, 
                              label_BIO_ids = label_BIO_ids,
                              label_CRO_ids = label_CRO_ids,
                              label_SRL_ids=label_SRL_ids))
    
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
