#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-25

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
sys.path.append('..')
from utils.config import opt
from models.Attention import SelfAttentionLayer


class Text_RCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_size, hidden_size, num_classes, dropout=.0,
                 num_layers=2, bidirectional=True, pretrained_emb=None, pooling='max'):
        super(Text_RCNN, self).__init__()
        self.model_name = 'Text_RCNN'
        self.encoder_size = encoder_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # initialize with pretrained
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        
        if bidirectional:
            self.n_cells = 2*num_layers
        else:
            self.n_cells = 1*num_layers

        if pooling == 'mean':
            RCNN_Pool = nn.AvgPool1d(kernel_size=opt.max_len)
        elif pooling == 'attn':
            RCNN_Pool = SelfAttentionLayer(hidden_size=2*encoder_size)
        else:
            RCNN_Pool = nn.MaxPool1d(kernel_size=opt.max_len)
        self.RCNN_Pool = RCNN_Pool
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=encoder_size, num_layers=num_layers,
                            bidirectional=bidirectional)
        self.cnin_dim = 2*encoder_size + embed_size
        self.cnout_dim = 2*encoder_size
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.cnin_dim, out_channels=self.cnout_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnout_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.cnout_dim, out_channels=self.cnout_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnout_dim),
            nn.ReLU(inplace=True),
            RCNN_Pool
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.cnout_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs, input_length, pooling='max'):
        input_length, prem_idx = input_length.sort(0, descending=True)
        inputs = inputs[prem_idx]

        embedded = self.embedding(inputs)  # batch x seq x dim
        embedded = embedded.permute(1, 0, 2)  # seq x batch x dim

        batch_size = embedded.size()[1]
        state_shape = self.n_cells, batch_size, self.encoder_size
        
        h0 = c0 = Variable(embedded.data.new(*state_shape).zero_())
        if opt.use_gpu:
            h0, c0 = h0.cuda(), c0.cuda()
        
        packed_input = pack_padded_sequence(embedded, input_length.cpu().numpy())
        packed_output, (ht, cit) = self.lstm(packed_input, (h0, c0))
        outputs, _ = pad_packed_sequence(packed_output)
        pad_len = embedded.size(0)-outputs.size(0)
        outputs = F.pad(outputs, (0,0,0,0,0,pad_len))

        text_cnin = torch.cat((outputs, embedded), dim=2)
        text_encoder = self.conv(text_cnin.permute(1, 2, 0))

        inverse_idx = torch.zeros(prem_idx.size()[0]).long()
        if opt.use_gpu:
            inverse_idx = inverse_idx.cuda()
        for i in range(prem_idx.size()[0]):
            inverse_idx[prem_idx[i]] = i
        text_encoder = text_encoder[inverse_idx]
        
        text_reshape = text_encoder.view(text_encoder.size(0), -1)
        text_drop = self.dropout(text_reshape)
        text_out = self.fc(text_drop)
        return text_out


if __name__ == '__main__':
    opt.use_gpu = False
    from torch.utils.data import DataLoader
    from scripts.data_loader import MyDataset
    model = Text_RCNN(vocab_size=opt.char_size, embed_size=opt.embed_dim,
                      encoder_size=opt.encoder_dim, hidden_size=opt.hidden_size,
                      num_classes=opt.num_classes, dropout=0.2, pooling='max')
    dataset = MyDataset(train=False)
    train_loader =  DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    for dd in train_loader:
        f,l,t = dd
        pred = model(f, l)
        print pred
        break


