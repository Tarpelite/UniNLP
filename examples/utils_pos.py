
from __future__ import absolute_import, division, print_function

import logging
import os
from io import open

logger = logging.getLogger(__name__)

class InputExample(object):

    def __init__(self, guid, words, labels):

        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}".txt.format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f.readlines():
            if line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), 
                                                words=words, 
                                                labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[1])
                tags.append(line[3])
    
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
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
    
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    cnt_counts = []
    last_tokens = []
    last_label_ids = []
    for(ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if ex_index == 0:
            last_tokens = tokens[-64:]
            last_label_ids = label_ids[-64:]
        
        else:
            tokens = last_tokens + tokens
            label_ids = last_label_ids + label_ids
            last_tokens= tokens[-64:]
            last_label_ids = label_ids[-64:]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        cnt_counts.append(len(tokens))
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

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
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    
    logger.info("*** Statistics ***")
    logger.info("*** max_len:{}  min_len:{} avg_len:{}***".format(max(cnt_counts), min(cnt_counts), sum(cnt_counts) / len(cnt_counts)))

    return features

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "X" not in labels:
            labels = ["X"] + labels
        return labels
    else:
        return ["ADJ", "ADP", "ADV", "AUX", "CONJ",
                "DET", "INTJ", "NOUN", "NUM", "PART", 
                "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", 
                "VERB", "X"]

            



