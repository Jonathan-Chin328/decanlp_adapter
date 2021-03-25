import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch.nn as nn

class Adapter(nn.Module):

    def __init__(self, dimension, hidden, activation='relu', dropout=0.2, layer_normalization=False):
        super().__init__()
        torch.manual_seed(0)
        self.down_feedforward = Feedforward(dimension, hidden)
        self.activation = getattr(F, activation)
        torch.manual_seed(1)
        self.up_feedforward = Feedforward(hidden, dimension)
        self.dropout = nn.Dropout(dropout)
        if layer_normalization is True:
            self.layer_norm = nn.LayerNorm(dimension)
        else:
            self.layer_norm = None

    def forward(self, x):
        if self.layer_norm is not None:
            # print('Adapter', x.shape)
            x = self.layer_norm(x)
            # print('layer_norm', bottle_neck.shape)
            # assert 1 == 2
        bottle_neck = self.activation(self.down_feedforward(x))
        return self.dropout(self.up_feedforward(bottle_neck)) + x


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.2):
        super().__init__()
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)
        for name, param in self.linear.named_parameters():
            param.data.fill_(float(1e-6))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))