import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import os
import sys
import numpy as np
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_sequence
from .adapter import Adapter


INF = 1e10
EPSILON = 1e-10

class LSTMDecoder(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(LSTMDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            input = self.dropout(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0., x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.new_ones((x.size(0), abs(y_len - x_len)))
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio, adapter=[], adapter_size=None):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)
        if 'transformer' in adapter:
            self.adapter = nn.ModuleList([Adapter(d_model, adapter_size) for i in range(10)])
        else:
            self.adapter = None


    def forward(self, *x, task_id=None, padding=None):
        if self.adapter is not None and task_id is not None:
            hidden = self.layer(*x, padding=padding)
            # key-value
            if isinstance(task_id, int):                    
                for i in range(len(self.adapter)):
                    if i != task_id:
                        for name, param in self.adapter[i].named_parameters():
                            param.requires_grad = False
                    else:
                        for name, param in self.adapter[i].named_parameters():
                            param.requires_grad = True
                return self.layernorm(x[0] + self.dropout(self.adapter[task_id](hidden)))
            # classfication
            else:
                adapter_output = torch.zeros(hidden.shape).to(hidden.device)
                for i in range(task_id.shape[-1]):
                    distribution_i = torch.reshape(task_id[:,i], (task_id.shape[0], 1, 1))
                    adapter_output += self.adapter[i](hidden) * distribution_i
                return self.layernorm(x[0] + self.dropout(adapter_output))
        else:
            return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))


class Attention(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.new_ones((key.size(1), key.size(1))).triu(1) * INF
            dot_products.sub_(tri.unsqueeze(0))
        if not padding is None:
            dot_products.masked_fill_(padding.unsqueeze(1).expand_as(dot_products), -INF)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, adapter, adapter_size):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout, adapter, adapter_size)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout, adapter, adapter_size)

    def forward(self, x, task_id, padding=None):
        return self.feedforward(self.selfattn(x, x, x, task_id=task_id, padding=padding), task_id=task_id)


class TransformerEncoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, adapter=[], adapter_size=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dimension, n_heads, hidden, dropout, adapter, adapter_size) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, task_id=None, padding=None):
        x = self.dropout(x)
        encoding = [x]
        for layer in self.layers:
            x = layer(x, task_id, padding=padding)
            encoding.append(x)
        return encoding


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, context_padding=None, answer_padding=None):
        x = self.selfattn(x, x, x, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding[1:]):
            x = layer(x, enc, context_padding=context_padding, answer_padding=answer_padding)
        return x


def mask(targets, out, squash=True, pad_idx=1):
    mask = (targets != pad_idx)
    out_mask = mask.unsqueeze(-1).expand_as(out).contiguous()
    out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    targets_after = targets[mask]
    return out_after, targets_after


class Highway(torch.nn.Module):
    def __init__(self, d_in, activation='relu', n_layers=1):
        super(Highway, self).__init__()
        self.d_in = d_in
        self._layers = torch.nn.ModuleList([Linear(d_in, 2 * d_in) for _ in range(n_layers)])
        for layer in self._layers:
            layer.bias[d_in:].fill_(1)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.d_in] if projected_input.dim() == 2 else projected_input[:, :, :self.d_in]
            nonlinear_part = self.activation(nonlinear_part)
            gate = projected_input[:, self.d_in:(2 * self.d_in)] if projected_input.dim() == 2 else projected_input[:, :, self.d_in:(2 * self.d_in)]
            gate = F.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LinearFeedforward(nn.Module):

    def __init__(self, d_in, d_hid, d_out, activation='relu'):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))

class PackedLSTM(nn.Module):

    def __init__(self, d_in, d_out, bidirectional=False, num_layers=1, 
        dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            d_out = d_out // 2
        self.rnn = nn.LSTM(d_in, d_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, inputs, lengths, hidden=None, silent=True):
        lens, indices = torch.sort(inputs.new_tensor(lengths, dtype=torch.long), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices] 
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(), 
            batch_first=self.batch_first), hidden)

        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))


class Embedding(nn.Module):

    def __init__(self, field, trained_dimension, dropout=0.0, project=True):
        super().__init__()
        self.field = field
        self.project = project
        dimension = 0
        pretrained_dimension = field.vocab.vectors.size(-1)
        self.pretrained_embeddings = [nn.Embedding(len(field.vocab), pretrained_dimension)]
        self.pretrained_embeddings[0].weight.data = field.vocab.vectors
        self.pretrained_embeddings[0].weight.requires_grad = False
        dimension += pretrained_dimension
        if self.project:
            self.projection = Feedforward(dimension, trained_dimension)
        dimension = trained_dimension
        self.dropout = nn.Dropout(dropout)
        self.dimension = dimension

    def forward(self, x, lengths=None, device=-1):
        pretrained_embeddings = self.pretrained_embeddings[0](x.cpu()).to(x.device).detach()
        return self.projection(pretrained_embeddings) if self.project else pretrained_embeddings

    def set_embeddings(self, w):
        self.pretrained_embeddings[0].weight.data = w
        self.pretrained_embeddings[0].weight.requires_grad = False


class SemanticFusionUnit(nn.Module):
 
    def __init__(self, d, l):
        super().__init__()
        self.r_hat = Feedforward(d*l, d, 'tanh')
        self.g = Feedforward(d*l, d, 'sigmoid')
        self.dropout = nn.Dropout(0.2)
 
    def forward(self, x):
        c = self.dropout(torch.cat(x, -1))
        r_hat = self.r_hat(c)
        g = self.g(c)
        o = g * r_hat + (1 - g) * x[0]
        return o


class LSTMDecoderAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context):
        if not self.dot:
            targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        else:
            targetT = input.unsqueeze(2)

        context_scores = torch.bmm(context, targetT).squeeze(2)
        context_scores.masked_fill_(self.context_mask, -float('inf'))
        context_attention = F.softmax(context_scores, dim=-1) + EPSILON
        context_alignment = torch.bmm(context_attention.unsqueeze(1), context).squeeze(1)

        combined_representation = torch.cat([input, context_alignment], 1)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_attention, context_alignment


class CoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding): 
        context_padding = torch.cat([context.new_zeros((context.size(0), 1), dtype=torch.long)==1, context_padding], 1)
        question_padding = torch.cat([question.new_zeros((question.size(0), 1), dtype=torch.long)==1, question_padding], 1)

        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat([context_sentinel, self.dropout(context)], 1) # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1) # batch_size x (question_length + 1) x features
        question = torch.tanh(self.proj(question)) # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1,2)) # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_context = self.normalize(affinity, context_padding) # batch_size x (context_length + 1) x 1
        attn_over_question = self.normalize(affinity.transpose(1,2), question_padding) # batch_size x (question_length + 1) x 1
        sum_of_context = self.attn(attn_over_context, context) # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question) # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context) # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question) # batch_size x (question_length + 1) x features
        return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], torch.cat([coattn_question, sum_of_context], 2)[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)

class QuestionClassifier(nn.Module):
    def __init__(self, d_in, d_out, max_question_length, bidirectional=False, num_layers=1, 
        dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            rnn_out = d_out // 2
        else:
            rnn_out = d_out
        self.rnn = nn.LSTM(d_in, rnn_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first
        self.max_question_length = max_question_length + 2   # for padding
        self.fc_layers = nn.Sequential(
            nn.Linear(d_out * self.max_question_length, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs, lengths, hidden=None, silent=True):
        lens, indices = torch.sort(inputs.new_tensor(lengths, dtype=torch.long), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices] 
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(), 
            batch_first=self.batch_first), hidden)
        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]

        # padding
        (batch_size, seq_len) = (outputs.shape[0], outputs.shape[1]) if self.batch_first else (outputs.shape[1], outputs.shape[0])
        pad = torch.zeros((batch_size, self.max_question_length - seq_len, outputs.shape[-1])).to(outputs.device)
        outputs = torch.cat((outputs, pad), 1)
        outputs = torch.reshape(outputs, (batch_size, self.max_question_length * outputs.shape[-1]))
        outputs = self.fc_layers(outputs)
        
        return outputs


class QuestionTrnasformerClassifier(nn.Module):
    def __init__(self, d_in, d_out, max_question_length, head=4, hidden=150, num_layers=2, 
        dropout=0.2, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        self.d_in = d_in
        self.transformer = TransformerEncoder(d_in, head, hidden, num_layers, dropout, adapter=[], adapter_size=None)
        self.batch_first = batch_first
        self.max_question_length = max_question_length
        self.fc_layers = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs, lengths, question_padding):
        if self.batch_first is True:
            batch_size = inputs.shape[0]
            outputs = self.transformer(inputs, task_id=None, padding=question_padding)
            # print(len(outputs))
            # print(outputs[-1].shape)
            # print(outputs[-1][:,0,:].shape)
            outputs = outputs[-1][:,0,:]
            outputs = self.fc_layers(outputs)
        else:
            assert self.batch_first is False

        return outputs

