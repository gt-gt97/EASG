import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
# from model.attention import MultiHeadAttention, SelfAttention, BiLinearAttn


class LockedDropout(nn.Module):
    # Direct from original code
    # Actually I don't understand why ..
    def __init__(self, dropout):
        super().__init__()
        # 设置RNN模型的dropout。
        self.dropout = dropout
    
    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderLSTM(nn.Module):
    # EncoderLSTM(in_dim, opt['hidden_dim'] // 2, 1, True, True, opt['rnn_dropout'], False, True)
    # num_units: 输出向量的维度等于隐藏节点数
    # dropout非0 ，那么除了最后一层输出，中间的隐含层（这里设置的nlayers=1层）就会进行丢失。需要把模型设置为train()。
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last, require_h=False):
        super().__init__()
        # 不清楚啥意思
        self.require_h = require_h
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                # 输入数据的特征维数
                input_size_ = input_size
                # opt['hidden_dim'] // 2
                output_size_ = num_units
            else:
                # 如果有多层，第一层输入是140（100+20+20），输出是128，第2层输入是256（双向LSTM）或者128（单向LSTM）输出是128（维度）
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            # 每层都使用一个rnn，参数为（输入数据的特征维数，LSTM中隐层的维度，循环神经网络的层数，是否使用双向LSTM
            # batch_first为ture表示模型的输入在内存中存储时，先存储第一个sequence，再存储第二个.按序列顺序存储）
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        
        '''
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        '''
        # 丢弃神经元的比例（0-1）。
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last
        
        # self.reset_parameters()
    
    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()
    
    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    # input是文档中每各字（词）的嵌入，并不是实体嵌入。（bz * doc_len * 140）
    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        # 丢弃层层数进行遍历
        for i in range(self.nlayers):
            # hidden, c = self.get_init(bsz, i)
            
            output = self.dropout(output)
            if input_lengths is not None:
                # pack_padded_sequence：按列进行压缩，原来填充的 PAD（一般初始化为0）占位符被删掉了。
                # 如果batch_first=True的话，那么相应的 input size 就是 (bsz×序列长度×每个字的维度)
                output = rnn.pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=False)
            
            # output, hidden = self.rnns[i](output, (hidden, c))
            # flatten_parameters：把你所有的weight压缩到一个连续的内存chuck中。
            self.rnns[i].flatten_parameters()
            # 多余的pad符号，这样会导致LSTM对它的表示通过了非常多无用的字符，这样得到的句子表示就会有误差,因此需要压缩padded去除多余的填充字符。
            output, hidden = self.rnns[i](output)

            if input_lengths is not None:
                # pad_packed_sequence：把压缩的对象再填充回去。
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                # 这里为什么会不一样？
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.require_h:
            if self.concat:
                return torch.cat(outputs, dim=2), hidden[0].permute(1, 0, 2).contiguous().view(bsz, -1)

        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]





