#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: FuYuwen
# Created Time: 2018-10-23

import codecs
import sys
sys.path.append('..')
from utils.config import opt
import os
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import pickle
from utils.text_process import uniform
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')


def get_category_num():
    labels = dict()
    with codecs.open(root_path+opt.orig_data_path, 'r')as f:
        for line in f:
            items = line.strip().split('\t')
            ll = items[1]
            labels[ll] = labels.get(ll, 0)+1
    label2idx = dict()
    for i, (ll, _) in enumerate(labels.items()):
        label2idx[ll] = i
    with open(root_path+opt.label_idx_path, 'w')as f:
        pickle.dump(label2idx, f)
    print label2idx


def csv_to_fast():
    fw = codecs.open(root_path+opt.fast_data_path, 'w')
    with open(root_path+opt.label_idx_path, 'r')as f:
        label2idx = pickle.load(f)
    with codecs.open(root_path+opt.orig_data_path, 'r')as f:
        for line in f:
            ques, intent = line.strip().split('\t')
            label = label2idx[intent]
            ques_format = uniform(ques)
            text = '__label__%s\t%s\n' % (str(label), ' '.join(ques_format))
            fw.write(text)


def save_char_idx():
    char2tf = defaultdict(int)
    with codecs.open(root_path+opt.orig_data_path, 'r')as f:
        for line in f:
            ques, _ = line.strip().split('\t')
            ques_format = uniform(ques)
            for c in list(ques_format):
                char2tf[c] = char2tf.get(c, 0)+1
    i=0
    char2idx=dict()
    char2idx[u'xxPADxx'] = 0
    for c, tf in char2tf.items():
        if tf>2:
            i+=1
            char2idx[c]=i
    char2idx[u'xxOOVxx'] = i+1
    print len(char2idx)
    with open(root_path+opt.char_idx_path, 'w')as f:
        pickle.dump(char2idx, f)


def compute_sent_len():
    sent_len = []
    with codecs.open(root_path+opt.orig_data_path, 'r')as f:
        for line in f:
            ques, _ = line.strip().split('\t')
            ques_format = uniform(ques)
            sent_len.append(len(ques_format))
    print (1.0*sum(sent_len))/len(sent_len)


if __name__ == '__main__':
    compute_sent_len()
    #save_char_idx()
    #csv_to_fast()
    #get_category_num()

