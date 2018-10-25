# -*- coding: utf-8 -*-
import torch


class Config(object):
    coding = 'UTF-8'

    lr = 1e-3
    max_len = 60
    batch_size = 64
    char_size = 1483
    word_size = 1518
    embed_dim = 128

    encoder_dim = 100
    dropout = 0.5
    hidden_size = 128
    num_classes = 45

    num_layers = 1  # LSTM layers
    kernel_sizes = [1, 2, 3, 5]

    max_epoch = 18
    use_gpu = torch.cuda.is_available()
    orig_data_path = '/data/ques_intent.txt'
    fast_data_path = '/data/ques_intent_fast.txt'
    label_idx_path = '/data/label2idx.pkl'
    char_idx_path = '/data/char2idx.pkl'
    model_root_path = '/data/checkpoint/'

opt = Config()
