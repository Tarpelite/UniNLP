
from __future__ import absolute_import, division, print_function

import logging
import os
from io import open

logger = logging.getLogger(__name__)
tag_list = ["acl", "advcl", "advmod", "amod", "appos",
                "aux", "case", "cc", "ccomp", "clf", 
                "compound", "conj", "cop", "csubj", "dep",
                "det", "discourse", "dislocated", "expl", "fixed",
                "flat", "goeswith", "iobj", "list", "mark",
                "nmod", "nsubj", "nummod", "obj", "obl",
                "orphan", "parataxis", "punct", "reparandum", "root",
                "vocative", "xcomp"]

class InputExample(object):

    def __init__(self, guid, words, tags, heads):

        self.guid = guid
        self.words = words
        self.tags = tags
        self.heads = heads

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, tag_ids, head_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_ids = tag_ids
        self.head_ids = head_ids

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        tags = []
        heads = []
        for line in f.readlines():
            if line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), 
                                                words=words, 
                                                tags=tags,
                                                heads=heads))
                    guid_index += 1
                    words = []
                    tags = []
                    heads = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[1])
                tag = line[7]
                if ":" in tag:
                    tag = tag.split(":")[0]
                
                head = line[6]
                if tag == "_" or head == "_":
                    continue
                tags.append(tag)
                heads.append(head)
    
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         tags=tags,
                                         heads=heads))
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
    
    for(ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        tokens = []
        tag_ids = []
        head_ids = []
        # only use the first of all the tokens
        for word, tag, head in zip(example.words, example.tags, example.heads):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if tag in label_map:
                tag_id = label_map[tag]
            else:
                tag_id = pad_token_label_id
            tag_ids.extend([tag_id] + [pad_token_label_id] * (len(word_tokens) - 1))
            head_ids.extend([int(head)] + [pad_token_label_id]*(len(word_tokens) - 1))


        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        cnt_counts.append(len(tokens))
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            tag_ids = tag_ids[:(max_seq_length - special_tokens_count)]
            head_ids = head_ids[:(max_seq_length - special_tokens_count)]

       
        tokens += [sep_token]
        tag_ids += [pad_token_label_id]
        head_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            tag_ids += [pad_token_label_id]
            head_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            tag_ids += [pad_token_label_id]
            head_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            tag_ids = [pad_token_label_id] + tag_ids
            head_ids = [pad_token_label_id] + head_ids
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
            tag_ids = ([pad_token_label_id] * padding_length) + tag_ids
            head_ids = ([pad_token_label_id] * padding_length) + head_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            tag_ids += ([pad_token_label_id] * padding_length)
            head_ids += ([pad_token_label_id] * padding_length)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(tag_ids) == max_seq_length
        assert len(head_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("tag_ids: %s", " ".join([str(x) for x in tag_ids]))
            logger.info("head_ids: %s", " ".join([str(x) for x in head_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              tag_ids=tag_ids,
                              head_ids=head_ids))
    
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
        return ["acl", "advcl", "advmod", "amod", "appos",
                "aux", "case", "cc", "ccomp", "clf", 
                "compound", "conj", "cop", "csubj", "dep",
                "det", "discourse", "dislocated", "expl", "fixed",
                "flat", "goeswith", "iobj", "list", "mark",
                "nmod", "nsubj", "nummod", "obj", "obl",
                "orphan", "parataxis", "punct", "reparandum", "root",
                "vocative", "xcomp"]

            



