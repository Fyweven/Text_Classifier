#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-24

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """"""

    def __init__(self, hidden_size, hops=1):
        """"""
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.hops = hops

        self.W_w = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        self.b_w = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.U_w = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hops))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = .1
        self.W_w.data.uniform_(-stdv, stdv)
        self.U_w.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        """
        """
        # print('inputs', inputs)
        inputs = inputs.permute(0, 2, 1)
        M = torch.bmm(inputs, self.W_w.unsqueeze(0).expand(inputs.size(0), *self.W_w.size()))  # batch x len x h
        M += self.b_w

        M = F.tanh(M)

        U = torch.bmm(M, self.U_w.unsqueeze(0).expand(M.size(0), *self.U_w.size()))  # batch x len x hops
        alpha = F.softmax(U, dim=1)  # b x l x hops

        # b x hops x h
        return torch.bmm(alpha.permute(0, 2, 1), inputs).permute(1, 0, 2).squeeze(0)  # hops x b x h
