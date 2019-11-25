from biunilm.seq2seq_loader import *
from tqdm import *

ner_labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def ner_truncate_tokens_pair(tokens_a, tokens_b, ner_labels, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    trunc_flag = None
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
            trunc_flag = "a"

        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
            trunc_flag = "b"

        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
                trunc_flag = "a"
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
                trunc_flag = "b"
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
                trunc_flag = "a"
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
                trunc_flag = "b"
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
            if trunc_flag == "a":
                del ner_labels[0]
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
            if trunc_flag == "a":
                ner_labels.pop()
    return num_truncated_a, num_truncated_b


class CoNLL2003Dataset(torch.utils.data.Dataset):

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []
        with open(file_src, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        words_tk = []
                        labels_seq = []
                        assert len(words) == len(labels)
                        for i, word in enumerate(words):
                            word_tk = tokenizer.tokenize(word)
                            for tk in word_tk:
                                words_tk.append(tk)
                                labels_seq.append(labels[i])
                        if len(words_tk) > 0 and len(labels_seq) > 0:
                            self.ex_list.append([words_tk, labels_seq])
                            words = []
                            labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(ner_labels.index(splits[-1].replace("\n", "")))
                    else:
                        labels.append(ner_labels.index("O"))
            if words:
                words_tk = []
                labels_seq = []
                assert len(words) == len(labels)
                for i, word in enumerate(words):
                    word_tk = tokenizer.tokenize(word)
                    for tk in word_tk:
                        words_tk.append(tk)
                        labels_seq.append(labels[i])

                if len(words_tk) > 0 and len(labels_seq) > 0:
                    self.ex_list.append([words, labels])
                    words = []
                    labels = []

        
        print("Load {} documents ".format(len(self.ex_list)))
    
    def __len__(self):
        return len(self.ex_list)
    
    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance
    
    def __iter__(self):
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4CoNLL2003(Pipeline):
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, labels = instance
        tokens_b = ['[SEP]']

        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = ner_truncate_tokens_pair(tokens_a, tokens_b, ner_labels, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        label_ids = [0] + labels + [0] + [0]*len(tokens_b) + [0]

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        label_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx,
                    oracle_pos, oracle_weights, oracle_labels)

        return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, -1, self.task_idx, label_ids)
        


        
