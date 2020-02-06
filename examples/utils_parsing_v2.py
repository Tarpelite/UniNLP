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

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, heads, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            heads: list. The heads of each token.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.heads = heads
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_ids = head_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            inputs = line.strip().strip("\n").split("\t")
            words = inputs[0].strip().split()
            heads = inputs[1].strip().split()
            labels = inputs[2].strip().split()
            
           
            assert len(words) == len(labels)
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         heads=heads,
                                         labels=labels))
    return examples


def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenizer,
                                 label_list=None,
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

    label_map = {label:idx for idx, label in enumerate(label_list)}
    features = []
    cnt_counts = []
    # last_tokens = []
    # last_label_ids = []
    get_label_list = []
    skip_num = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        head_ids = []
        label_ids = []
        all_doc_tokens = []

        tok_to_orig_index = []
        orig_to_tok_index = []

        for word, head, label in zip(example.words, example.heads, example.labels):
            orig_to_tok_index.append(len(tokens))
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if head == '_' or int(head) > (max_seq_length -2) :
                head = pad_token_label_id  # 0 for [cls] and [ROOT]

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            head_ids.extend([int(head)] + [pad_token_label_id] * (len(word_tokens) - 1))
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            if label not in get_label_list: 
                get_label_list.append(label)


        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        cnt_counts.append(len(tokens))
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            # skip long sentence
            skip_num += 1
            continue
        
        # convert head to absolute ids in these sequence 
        orig_to_tok_index = [x+1 for x in orig_to_tok_index]
        new_head_ids = []
        
        for x in head_ids:
            if x == 0: # special token will be left for [0]
                new_head_ids += [0]
            elif x == pad_token_label_id:
                new_head_ids += [pad_token_label_id]
            else:
                new_head_ids += [orig_to_tok_index[x-1]]
      
        head_ids = new_head_ids

        tokens += [sep_token]
        head_ids += [pad_token_label_id]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            head_ids += [pad_token_label_id]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            head_ids += [pad_token_label_id]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            head_ids = [pad_token_label_id]  + head_ids
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
            head_ids = ([pad_token_label_id]*padding_length) + head_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            head_ids += ([pad_token_label_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(head_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("head_ids: %s", " ".join([str(x) for x in head_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              head_ids = head_ids,
                              label_ids=label_ids))
    
    logger.info("*** Statistics ***")
    logger.info("*** max_len:{}  min_len:{} avg_len:{}***".format(max(cnt_counts), min(cnt_counts), sum(cnt_counts) / len(cnt_counts)))
    logger.info(" skip {} long sentences".format(skip_num))
    print("dataset label list", get_label_list)
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        print("get_labels", labels)
        return labels
    else:
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
