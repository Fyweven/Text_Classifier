#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Author: FuYuwen
# Created Time: 2018-10-23

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.Attention import SelfAttentionLayer
from utils.config import opt


class Text_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_channels, kernel_sizes, hidden_size, num_classes,
                 dropout=.0, pretrained_emb=None, pooling='max'):
        super(Text_CNN, self).__init__()
        self.model_name = 'Text_CNN'

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
			self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        if pooling == 'mean':
            CNN_Pool = nn.AvgPool1d(kernel_size=opt.max_len)
        elif pooling == 'attn':
            CNN_Pool = SelfAttentionLayer(hidden_size=out_channels)
        else:
            CNN_Pool = nn.MaxPool1d(kernel_size=opt.max_len)

        self.text_convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv1d(in_channels=embed_dim, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
					nn.BatchNorm1d(out_channels),
					nn.ReLU(inplace=True),
                    CNN_Pool
					)
				for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
				nn.Linear(len(kernel_sizes)*(out_channels), hidden_size),
				nn.BatchNorm1d(hidden_size),
				nn.ReLU(inplace=True),
				nn.Linear(hidden_size, num_classes)
				)

    def forward(self, text):
		text = self.embedding(text)
		text_encoder = [text_conv(text.permute(0, 2, 1)) for text_conv in self.text_convs]
		text_encoder = torch.cat(text_encoder, dim=1)
		text_reshape = text_encoder.view(text_encoder.size(0), -1)
		text_drop = self.dropout(text_reshape)
		text_out = self.fc(text_drop)
		return text_out


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	from scripts.data_loader import MyDataset
	model = Text_CNN(vocab_size=opt.char_size, embed_dim=opt.embed_dim,
                  out_channels=opt.encoder_dim, kernel_sizes=opt.kernel_sizes,
                  hidden_size=opt.hidden_size, num_classes=opt.num_classes, pooling='attn')
	dataset = MyDataset(train=True)
	train_loader =  DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
	for dd in train_loader:
		f,l,t = dd
		pred = model(f)
		print pred
		break# Created Time: 2018-10-24
