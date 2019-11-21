""" Load UD dataset. """

from __future__ import absolute_import, division, whitespace_tokenize

import json
import logging
import math
import collections
from tqdm import *
from io import open

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

class UDExample(object):
    """
    """
    def __init__(self,
                UD_id,
                doc_tokens,
                pos_tags=None):
        self.UD_id = UD_id
        self.doc_tokens = doc_tokens
        self.pos_tags = pos_tags
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.UD_id)
        s += ", doc_tokens: %s" % (self.doc_tokens)
        if self.pos_tags is not None:
            s += ", pos_tags: %s"%(self.pos_tags)
        return s


class InputFeatures(object):

    def __init__(self, unique_id, 
                input_ids, 
                input_mask, segment_ids, label_ids):
        
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
    
def read_UD_examples(input_file, is_training):

    data = []
    instances = []
    tokens = []
    tags = []
    cnt = 0
    examples = []
    doc_tokens_lens = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(line)
    
    for line in tqdm(data):
        if line == "\n":
            if is_training:
                instances.append([tokens, tags])
            else:
                instances.append([tokens, None])
            tokens = []
            tags = []
        elif line.startswith("#"):
            pass
        else:
            line = line.strip("\n").split("\t")
            tokens.append(line[1])
            if is_training:
                tags.append(line[3])
    
    for i, instance in enumerate(instances):
        example =  UDExample(
            UD_id=i,
            doc_tokens = instance[0], 
            pos_tags = instance[2] 
        )
        examples.append(example)
        doc_tokens_lens.append(len(doc_tokens))
    
    print("Statistics")
    print("max_len: {} min_len:{} avg_len: {}".format(max(doc_tokens_lens), min(doc_tokens_lens), sum(doc_tokens_lens)/len(doc_tokens_lens)))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0, pad_token_label_id=-1,
                                 mask_padding_with_zero=True):
    
    unique_id = 1000000000

    features = []
    logger.info("converting {} examples to features".format(features))

    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ",
                "DET", "INTJ", "NOUN", "NUM", "PART", 
                "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", 
                "VERB", "X"]
    pos_tags_dict = {tag:i+1 for i, tag in enumerate(pos_tags)}

    for (example_index, example) in enumerate(tqdm(examples)):

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_pos_tags = []
        all_pos_tags_idx = []
        for(i, token) in enumerate(example.tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                if example.pos_tags is not None:
                    all_pos_tags.append(example.pos_tags[i])
        
        max_tokens_for_doc = max_seq_length - 3

        for tag in all_pos_tags:
            if tag in pos_tags_dict:
                all_pos_tags_idx.append(pos_tags_dict[tag])
            else:
                all_pos_tags_idx.append(0)

        if len(all_doc_tokens) > max_tokens_for_doc:
            tokens = all_doc_tokens[:max_tokens_for_doc]
            all_pos_tags_idx = all_pos_tags_idx[:max_tokens_for_doc]
        
        tokens = [cls_token] + all_doc_tokens 
        if is_training:
            label_ids = [17] + all_pos_tags_idx 
        
        tokens += ["[SEP]"]
        if is_training:
            label_ids += [17]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0]*len(input_ids)
        segment_ids = [sequence_a_segment_id]*len(tokens)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0]*padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        if is_training:
            label_ids += ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) ==  max_seq_length
        assert len(segment_ids) == max_seq_length
        if is_training:
            assert len(label_ids) == max_seq_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("UD_id: %s", example.UD_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            if is_training:
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        
        features.append(
            InputFeatures(input_ids=input_ids, 
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=label_ids if is_training else None, 
                            tokens=all_doc_tokens,
                         )
        )


        


        
            

    


        
