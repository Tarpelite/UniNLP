import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel


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
        nn.init.xavier_uniform(self.U)
    
    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)

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
        h0 = torch.zeros(self.num_layers*self.num_directions, batch, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*self.num_directions, batch, self.hidden_size).cuda()
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

        self.encoder = RecurrentEncoder(config.hidden_size, mlp_input, 1, batch_first=True, dropout=config.hidden_dropout_prob)
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
        self.critierion = nn.CrossEntropyLoss(ignore_index=-100, size_average=True)
    
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
        for head in heads:
            print(head)
        print(flag)
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





        


