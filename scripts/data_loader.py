#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-23

import sys
sys.path.append('..')
from utils.config import opt
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from scripts import char_feature
import os


def getTrainData():
    datas = char_feature.load_char_data()
    return datas

class MyDataset(data.Dataset):
    def __init__(self, train=True, dev_ratio=0.1):
        feat, lens, labels = getTrainData()
        self.train = train
        dev_index = -1 * int(dev_ratio * len(labels))
        self.train_feats, self.train_lens, self.train_labels = feat[:dev_index], lens[:dev_index], labels[:dev_index]

        self.test_feats, self.test_lens, self.test_labels = feat[dev_index:], lens[dev_index:], labels[dev_index:]

    def setTrain(self, train):
        self.train = train

    def __getitem__(self, index):
        if self.train:
            txt, _len, target = self.train_feats[index], self.train_lens[index], self.train_labels[index]
        else:
            txt, _len, target = self.test_feats[index], self.test_lens[index], self.test_labels[index]
        return txt, _len, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


if __name__ == '__main__':
    dataset = MyDataset(train=True)
    train_loader =  DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    print (len(train_loader))
    for dd in train_loader:
        f,l,t =  dd
        print type(f)
        print type(l)
        print type(t)
        break
