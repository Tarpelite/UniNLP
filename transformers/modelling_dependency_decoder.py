import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F


PAD_INDEX = nn.CrossEntropyLoss().ignore_index

class MLP(nn.Module):
    """Module for an MLP with dropout"""
    def __init__(self, input_size, layer_size, depth, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                    nn.Linear(input_size, layer_size))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                        act_fn(inplace=False))
            
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                        nn.Dropout(dropout))
            input_size = layer_size
    
    def forward(self, x):
        return self.layers(x)

class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        # nn.init.xavier_uniform_(self.U)
    
    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=384, hidden_size=384):
        super(BiLSTMEncoder, self).__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM( input_size, 
                            hidden_size, 
                            batch_first=True, 
                            bidirectional=True)
        self.num_directions=2
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).cuda()
        out, _ = self.rnn(x, (h0, c0))
        return out



class RecurrentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                batch_first, dropout):
        
        super(RecurrentEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.batch_first = batch_first

        self.rnn = getattr(nn, 'LSTM')(input_size, hidden_size, num_layers,
                                       batch_first=batch_first, 
                                       dropout=dropout, 
                                       bidirectional=True)
        
        
        self.train_hidden_init = False
    
    # def get_hidden(self, batch):
    #     args = self.num_layers*self.num_directions, batch, self.hidden_size
    #     h0 = torch.randn(*args)
    #     c0 = torch.randn(*args)
    #     h0, c0 = h0.cuda(), c0.cuda()
    #     return h0, c0
    
    def forward(self, x, lengths):
        batch = x.size(0) if self.batch_first else x.size(1)
        h0 = torch.zeros(self.num_layers*self.num_directions, batch, self.hidden_size).to(torch.device("cuda"))
        c0 = torch.zeros(self.num_layers*self.num_directions, batch, self.hidden_size).to(torch.device("cuda"))
        out, _ = self.rnn(x, (h0, c0))
        return out


class BiAffineParser(BertPreTrainedModel):
    """Biaffine Dependency Parser"""
    def __init__(self, config, mlp_input, mlp_arc_hidden,
                mlp_lab_hidden, mlp_dropout, 
                num_labels, critierion, max_len):
        super(BiAffineParser, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = RecurrentEncoder(config.hidden_size, mlp_input, lstm_layers, batch_first=True, dropout=0.1)
        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input*2, mlp_arc_hidden, 2, "ReLU", mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input*2, mlp_arc_hidden, 2, "ReLU", mlp_dropout)

        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input*2, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)
        self.lab_mlp_d = MLP(mlp_input*2, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = BiAffine(mlp_lab_hidden, num_labels)
        

        # Loss criterion
        self.critierion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, mask_qkv=None, task_idx=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        
        words = input_ids
        aux = (words != PAD_INDEX).long().sum(-1)

        x = sequence_output

        h = self.encoder(x, aux)

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)
        lab_h = self.lab_mlp_h(h)
        lab_d = self.lab_mlp_d(h)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)
        return S_arc, S_lab
    
    def arc_loss(self, S_arc, heads):
        S_arc = S_arc.transpose(-1, -2)
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))
        heads = heads.view(-1)
        # print("heads", heads)
        # print("check heads")
        # print("S_arc", S_arc.shape)
        # print("heads", heads.shape)
        flag = True
        # for head in heads:
        #     print(head)
        # print(flag)
        return self.critierion(S_arc, heads)
    
    def lab_loss(self, S_lab, heads, labels):
        heads = heads.unsqueeze(1).unsqueeze(2)              # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)      # [batch, n_labels, 1, sent_len]
        # print("heads", heads.shape)
        # print("S_lab", S_lab.shape)
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)     # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)                      # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
        labels = labels.view(-1)                             # [batch*sent_len]
        return self.critierion(S_lab, labels)

def LocalCELoss(inputs, labels):
    # inputs shape [batch, max_len, max_len]
    # labels [batch, max_len]
    inputs = torch.exp(inputs)
    pass

class BertForDependencyParsing(BertPreTrainedModel):
    def __init__(self, config, num_labels=20):
        super(BertForDependencyParsing, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.bilstm = BiLSTMEncoder(input_size=config.hidden_size, hidden_size=config.hidden_size)

        self.arc_mlp_head = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.arc_mlp_dep = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

        self.label_mlp_head = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.label_mlp_dep = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

        self.arc_biaffine = BiAffine(2*config.hidden_size, 1)
        self.lab_biaffine = BiAffine(2*config.hidden_size, num_labels)
    
    def forward(self, 
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                heads=None,
                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        r = self.bilstm(sequence_output)
        # print("r.shape", r.shape)
        # print("self.arc_mlp_head", self.arc_mlp_head)
        arc_head = self.arc_mlp_head(r)
        arc_dep = self.arc_mlp_dep(r)

        label_head = self.label_mlp_head(r)
        label_dep = self.label_mlp_head(r)

        s_arc = self.arc_biaffine(arc_head, arc_dep)
        s_lab = self.lab_biaffine(label_head, label_dep) # [batch, num_labels, seq_len, seq_len]

        outputs = (s_arc, s_lab)
        loss_func = nn.CrossEntropyLoss()
        if heads is not None and labels is not None:
            s_arc = s_arc.contiguous().view(-1, s_arc.size(-1))
            heads = heads.view(-1)
            # print("s_arc shape", s_arc.shape)
            # print("heads shape", heads.shape)
            # print(s_arc)
            # print(heads)
            arc_loss = loss_func(s_arc, heads)
            label_loss = arc_loss
            # print("s_lab", s_lab.shape)
            # heads = heads.unsqueeze(1).unsqueeze(2)              # [batch, 1, 1, sent_len]
            # heads = heads.expand(-1, s_lab.size(1), -1, -1)      # [batch, n_labels, 1, sent_len]
            # # print("heads", heads.shape)
            # # print("S_lab", S_lab.shape)
            # s_lab = torch.gather(s_lab, 2, heads).squeeze(2)     # [batch, n_labels, sent_len]
            # s_lab = s_lab.transpose(-1, -2)                      # [batch, sent_len, n_labels]
            # s_lab = s_lab.contiguous().view(-1, s_lab.size(-1))  # [batch*sent_len, n_labels]
            # labels = labels.view(-1)                             # [batch*sent_len]
            # label_loss = loss_func(s_lab, labels)
            outputs = (arc_loss, label_loss) + outputs
        return outputs

class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()

        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):

        # input1 size: [batch_size, seq_len, feature_size]
        # input1.new_ones [batch_size, seq_len, 1]
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)

        return self.W_binin(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, dropout=0):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)

        self.hidden_func = F.relu

        self.scorer = BiaffineScorer(hidden_size, hidden_size, outpout_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))

class BiaffineDependencyModel(BertPreTrainedModel):
    def __init__(self, config, num_labels, biaffine_hidden_size):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = num_labels
        self.unlabeled_biaffine = DeepBiaffineScoerer(config.hidden_size,
                                                      config.hidden_size,
                                                      biaffine_hidden_size,
                                                      1, dropout=0.1)
        
        self.labeled_biaffine = DeepBiaffineScorer(config.hidden_size,
                                                   config.hidden_size,
                                                   biaffine_hidden_size,
                                                   num_labels,
                                                   dropout=0.1)
    

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                heads=None,
                labels=None):
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        unlabeled_scores = self.unlabeled_biaffine(sequence_output, sequence_output).squeeze(3)
        labeled_scores = self.labeled_biaffine(sequence_output, sequence_output)

        loss = None
        outputs = (unlabeled_scores, labeled_scores) + outputs
        if heads is not None:
            # do mask weights
            weights = torch.ones(word_pad_mask.size(0), inputs_ids.size(1), inputs_ids.size(1),
                                    dtype=unlabeled_sccores.dtype, device=unlabeled_scores.device)
            input_mask = torch.eq(heads, -100)
            weights = weights.maked_fill(input_mask, 0)
            weights = weights.maked_fill(input_mask, 0)

            # words_num = torch.sum(torch.ge(input_ids, 0)).item()

            loss_fct = nn.BCEWithLogitsLoss(weight=weights, reduction="sum")
            dep_arc_loss = loss_fct(unlabeled_scores, heads)
            if loss is None:
                loss = dep_arc_loss
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            labeled_scores = labeled_scores.contiguous().view(-1, self.num_labels)
            dep_label_loss = loss_fct(labeled_scores, labels.view(-1))
            if loss is None:
                loss = dep_label_loss
            else:
                loss += dep_label_loss
        
        if loss is not None:
            outputs = (loss, ) + outputs
        return outputs




