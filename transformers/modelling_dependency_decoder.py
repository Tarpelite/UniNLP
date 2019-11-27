import torch
import torch.nn as nn
from modelling_bert import BertModel, BertPreTrainedModel

class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first
    
    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x 
    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask

class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        # [batch_size, 1, seq_len, d]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, d]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        s = x @ self.weight @ y.transpose(-1, -2)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class DecoderModel(BertPreTrainedModel):

    def __init__(self, args):
        super(DecoderModel, self).__init__()
        self.args = args
        self.config = args.config
        self.bert = BertModel(args.config)
        self.embed_dropout = IndependentDropout(args.embed_dropout)

        
        # the MLP layer
        self.mlp_arc_h = MLP(n_in=self.config.hidden_size,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.config.hidden_size,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.config.hidden_size,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.config.hidden_size,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        
        # the Biaffine layers

        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                bias_x=True, bias_y=False)
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                bias_x=True, bias_y=True)
        
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index
    

         
