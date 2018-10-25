#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Author: FuYuwen
# Created Time: 2018-10-24

import torch
from torch import nn, optim
import sys
sys.path.append('..')
from utils.config import opt
from scripts import data_loader
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from models.Text_CNN import Text_CNN
from models.CNN_Inception import CNN_Inception
import os
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def train():
    dataset = data_loader.MyDataset(train=True)
    print ('数据加载完成！')

    #model = Text_CNN(vocab_size=opt.char_size, embed_dim=opt.embed_dim,
    #                 out_channels=opt.encoder_dim, kernel_sizes=opt.kernel_sizes,
    #                 hidden_size=opt.hidden_size, num_classes=opt.num_classes, dropout=0.3,
    #                 pooling='attn')
    model = CNN_Inception(vocab_size=opt.char_size, embed_dim=opt.embed_dim, encoder_dim=opt.encoder_dim, hidden_size=opt.hidden_size, num_classes=opt.num_classes, pooling='max')

    if opt.use_gpu:
        print 'use_gpu'
        model.cuda()

    initNetParams(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    final_P = 0.0
    final_epo = 0.0

    for epoch in range(1, opt.max_epoch+1):
        print ('')
        print('*' * 6 + '[epoch {}/{}]'.format(epoch,opt.max_epoch) +'*' * 6)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        dataset.setTrain(True)
        train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        step = 0
        time.sleep(0.1)
        for data in tqdm(train_loader):
            txt, _, label = data
            txt = Variable(txt)
            label = Variable(label)
            if opt.use_gpu:
                txt, label = txt.cuda(), label.cuda()
            optimizer.zero_grad()
            out = model(txt)
            loss = criterion(out,label)
            running_loss += loss.data.mean() * label.size(0)
            running_acc += get_acc(out,label)

            loss.backward()
            optimizer.step()

            step+=1
            if step % 40 == 0:
                print('Loss: {:.6f}, Acc: {:.6f}'.format(running_loss / (opt.batch_size * step), running_acc / (opt.batch_size * step)))
        print ('')

        P = eval(model, dataset)
        if P > final_P:
            torch.save(model.state_dict(),
                       root_path + opt.model_root_path + '%s_best.pkl' % (model.model_name))
            final_P = P
            final_epo = epoch
    print  (final_epo, final_P)


def eval(model, dataset):
    model.eval()
    dataset.setTrain(False)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    eval_acc = 0.0
    test_num = 0
    print('*' * 3 + 'eval' + '*' * 3)
    time.sleep(0.1)
    for data in tqdm(test_loader):
        txt, _, label = data
        txt = Variable(txt)
        label = Variable(label)
        if opt.use_gpu:
            txt = txt.cuda()
            label = label.cuda()
        out = model(txt)
        eval_acc += get_acc(out, label)
        test_num += label.size(0)
    print('Eval Acc: {}'.format(eval_acc / test_num))
    return eval_acc / test_num


def get_acc(out, label):
    pred = torch.argmax(out, 1)
    num_correct = (pred == label).sum()
    return num_correct.item()


def initNetParams(net):
    '''''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            init.constant_(m.bias, 0)


if __name__ == '__main__':
    if opt.use_gpu:
        torch.cuda.set_device(5)
    train()
