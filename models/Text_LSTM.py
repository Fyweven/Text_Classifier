#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-24

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
sys.path.append('..')
from utils.config import opt
from models.Attention import SelfAttentionLayer


class Text_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_size, hidden_size, num_classes, dropout=.0, bidirectional=True, pretrained_emb=None, pooling='max'):
        super(Text_LSTM, self).__init__()
        self.model_name = 'Text_LSTM'
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.birnn = bidirectional

        if bidirectional:
            self.n_cells = 2
        else:
            self.n_cells = 1

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # initialize with pretrained
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            
        if pooling == 'mean':
            LSTM_Pool = nn.AvgPool1d(kernel_size=opt.max_len)
        elif pooling == 'attn':
            LSTM_Pool = SelfAttentionLayer(hidden_size=2*encoder_size)
        elif pooling == 'max':
            LSTM_Pool = nn.MaxPool1d(kernel_size=opt.max_len)
        else:
			LSTM_Pool = None
        self.LSTM_Pool = LSTM_Pool

        self.encoder = nn.LSTM(embed_size, encoder_size, bidirectional=self.birnn, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
				nn.Linear(2*encoder_size, hidden_size),
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
        packed_output, (ht, cit) = self.encoder(packed_input, (h0, c0))
        outputs, _ = pad_packed_sequence(packed_output)

        if self.LSTM_Pool is None:
            text_encoder = ht[-1] if not self.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        else:
            outputs = outputs.permute(1, 2, 0)
            text_encoder = self.LSTM_Pool(outputs)
        
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
    model = Text_LSTM(vocab_size=opt.char_size, embed_size=opt.embed_dim,
                      encoder_size=opt.encoder_dim, hidden_size=opt.hidden_size,
                      num_classes=opt.num_classes, dropout=0.2, pooling='last')
    dataset = MyDataset(train=True)
    train_loader =  DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    for dd in train_loader:
		f,l,t = dd
		pred = model(f, l)
		print pred
		break


