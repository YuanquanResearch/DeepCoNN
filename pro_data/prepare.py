#!/bin/usr/env python
# -*- coding: utf-8 -*-
__date__ = '5/15/18 16:55'

import pandas as pd
import numpy as np
import os
# import pickle
import cPickle as pickle

datafile = "Movies"
# datafile = "Movies"

def get_id(file, tofile):
    df = pd.read_csv(file, header=None)  # u,i,r,t

    user_id = np.array(df[0])
    item_id = np.array(df[1])
    y = np.array(df[2])

    user_id = user_id[:, np.newaxis]
    item_id = item_id[:, np.newaxis]
    y = y[:, np.newaxis]

    batches = list(zip(user_id, item_id, y))
    output = open(tofile, 'wb')
    pickle.dump(batches, output)

    return dict(zip(zip(df[0], df[1]), df.index))

# trainset 里面的{(user, item):seq_ix}
train_ui = get_id('../../RRN/data/%s/train.csv' % datafile, '../data/%s/train' % datafile)
val_ui = get_id('../../RRN/data/%s/val.csv' % datafile, '../data/%s/val' % datafile)
test_ui = get_id('../../RRN/data/%s/test.csv' % datafile, '../data/%s/test' % datafile)



def get_uitext(file, tofile):
    df = pd.read_csv(file)  # u,i,r,t, wordid0, wordid1,...
    # df = df.sample(frac=1).reset_index(drop=True)  # shuffle row

    u_text = {}
    i_text = {}

    vocabulary_user = set()  # 记录user的word个数 应该是20000

    print "read in.csv begin"
    for ix, row in df.iterrows():
        if ix % 10000 == 0:
            print ix

        user = row['user']
        item = row['item']

        # val, test 过滤掉
        if (user, item) not in train_ui:
            continue

        if user not in u_text:
            u_text[user] = []

        if item not in i_text:
            i_text[item] = []


        # 对于每个review, 去掉之前填充的0
        for wordid in row[4:]:
            vocabulary_user.add(wordid)
            if wordid != 0:
                u_text[user].append(wordid)
                i_text[item].append(wordid)

    print "read in.csv end"

    u = np.array([len(x) for x in u_text.itervalues()])
    x = np.sort(u)
    u_len = x[int(0.70 * len(u)) - 1]  # 每个user的reviews document的总长度排序 取第0.85长的长度

    i = np.array([len(x) for x in i_text.itervalues()])
    y = np.sort(i)
    i_len = y[int(0.70 * len(i)) - 1]  # 每个item的reviews document的总长度排序 取第0.85长的长度

    # u_len = max([len(x) for x in u_text.values()])  # 一个user最长的word list
    # i_len = max([len(x) for x in i_text.values()])

    print "u_len %d, i_len %d" % (u_len, i_len)
    # exit(0)

    # 填充
    for u in u_text:
        if len(u_text[u]) > u_len:
            u_text[u] = np.array(u_text[u][:u_len])
        else:
            u_text[u] = np.array([0]*(u_len-len(u_text[u])) + u_text[u])

    for i in i_text:
        if len(i_text[i]) > i_len:
            i_text[i] = np.array(i_text[i][:i_len])
        else:
            i_text[i] = np.array([0]*(i_len-len(i_text[i])) + i_text[i])

    para = {}
    para['user_num'] = len(set([x[0] for x in train_ui.keys()])) + 1
    para['item_num'] = len(set([x[1] for x in train_ui.keys()])) + 1
    para['user_length'] = u_text[1].shape[0]  # 每个user的reviews document中共有多少个word
    para['item_length'] = i_text[1].shape[0]
    para['user_vocab'] = len(vocabulary_user)
    para['item_vocab'] = len(vocabulary_user)
    para['train_length'] = len(train_ui)
    para['val_length'] = len(val_ui)
    para['test_length'] = len(test_ui)
    para['u_text'] = u_text
    para['i_text'] = i_text
    output = open(tofile, 'wb')

    print "user_num %d, item_num %d, u_len %d, i_len %d, user_vocab %d, train_len %d, val_len %d, test_len %d" % \
          (para['user_num'], para['item_num'], para['user_length'], para['item_length'], para['user_vocab'],
           para['train_length'], para['val_length'], para['test_length'])

    pickle.dump(para, output)

get_uitext('../../RRN/data/%s/in.csv' % datafile, '../data/%s/para' % datafile)