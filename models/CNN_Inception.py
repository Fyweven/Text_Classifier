#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-25

import torch
from torch import nn
import sys
sys.path.append('..')
from utils.config import opt
from models.Attention import SelfAttentionLayer
#知乎看山杯模型


class Inception(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, bn=True):
        super(Inception, self).__init__()
        assert out_dim%4 == 0
        out_ds = [out_dim/4]*4
        self.activate = nn.Sequential()
        if bn: self.activate.add_module('bn', nn.BatchNorm1d(out_dim))
        if relu: self.activate.add_module('relu', nn.ReLU(inplace=True))
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_dim, out_ds[0], 1, stride=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_dim, out_ds[1], 1),
            nn.BatchNorm1d(out_ds[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ds[1], out_ds[1], 3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_dim, out_ds[2], 3, padding=1),
            nn.BatchNorm1d(out_ds[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ds[2], out_ds[2], 5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_dim, out_ds[3], 3, stride=1, padding=1)
        )

    def forward(self, _input):
        out_b1 = self.branch1(_input)
        out_b2 = self.branch2(_input)
        out_b3 = self.branch3(_input)
        out_b4 = self.branch4(_input)
        _output = self.activate(torch.cat((out_b1,out_b2,out_b3,out_b4), dim=1))
        return _output


class CNN_Inception(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_size, num_classes,
                 dropout=.0, pretrained_emb=None, pooling='max'):
        super(CNN_Inception, self).__init__()
        self.model_name = 'CNN_Inception'

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.incept_dim = 4*encoder_dim
        if pooling == 'mean':
            Incept_Pool = nn.AvgPool1d(kernel_size=opt.max_len)
        elif pooling == 'attn':
            Incept_Pool = SelfAttentionLayer(hidden_size=self.incept_dim)
        else:
            Incept_Pool = nn.MaxPool1d(kernel_size=opt.max_len)

        self.incep_encoder = nn.Sequential(
            Inception(embed_dim, self.incept_dim),
            Inception(self.incept_dim, self.incept_dim),
            Incept_Pool
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.incept_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, text):
        text = self.embedding(text)
        text_encoder = self.incep_encoder(text.permute(0,2,1))
        text_reshape = text_encoder.view(text_encoder.size(0), -1)
        text_drop = self.dropout(text_reshape)
        text_out = self.fc(text_drop)
        return text_out


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	from scripts.data_loader import MyDataset
	model = CNN_Inception(vocab_size=opt.char_size, embed_dim=opt.embed_dim,
                       encoder_dim=opt.encoder_dim, hidden_size=opt.hidden_size,
                       num_classes=opt.num_classes, pooling='attn')
	dataset = MyDataset(train=True)
	train_loader =  DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
	for dd in train_loader:
		f,l,t = dd
		pred = model(f)
		print pred
		break

