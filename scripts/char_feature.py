#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-23

import torch
import pickle
import codecs
from tqdm import tqdm
import os
import sys
sys.path.append('..')
from utils.text_process import uniform
from utils.config import opt
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
reload(sys)
sys.setdefaultencoding('utf8')


def load_char_idx():
    print 'load char idx...'
    with open(root_path+opt.char_idx_path, 'r')as f:
        char2idx = pickle.load(f)
    return char2idx


def load_label_idx():
    print 'load label idx...'
    with open(root_path+opt.label_idx_path, 'r')as f:
        label2idx = pickle.load(f)
    return label2idx


def sent_to_idx(sent, c2i):
    sent = uniform(sent)
    cs = list(sent)
    sent_len = min(len(cs), opt.max_len)
    cs = cs[:sent_len]
    idx = [c2i.get(c, len(c2i)-1) for c in cs]
    for i in range(sent_len, opt.max_len):
        idx.append(0)
    return torch.LongTensor(idx), sent_len


def load_char_data():
    _data = []
    char2idx = load_char_idx()
    label2idx = load_label_idx()
    print 'load char data..'
    ques_feats, ques_lens, labels = [], [], []
    with codecs.open(root_path+opt.orig_data_path, 'r')as f:
        for line in tqdm(f):
            ques, intent = line.strip().split('\t')
            ques_idx, ques_len = sent_to_idx(ques, char2idx)
            ll = label2idx[intent]
            ques_feats.append(ques_idx)
            ques_lens.append(ques_len)
            labels.append(ll)
    return ques_feats, ques_lens, labels


if __name__ == '__main__':
	ll = load_label_idx()
	print len(ll)
	
	#load_char_data()
