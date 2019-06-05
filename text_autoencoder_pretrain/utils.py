#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_len = len(max(sents, key=len))
    for sent in sents:
        len_diff = max_len - len(sent)
        sents_padded.append(sent[:] + [pad_token]*len_diff)


    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['<end>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False, train=True, start=0):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    if train:  # checkpoint indices to prevent training interruption
        index_array = [0]
        current_index = [0]

        if "index_array" in os.listdir(os.getcwd()):
            index_array = pickle.load(open("index_array", "rb"))
            current_index = pickle.load(open("current_index", "rb"))
        if current_index[-1] != index_array[-1]:
            index_array = index_array[index_array.index(current_index[0]):]
            shuffle=False
            batch_num = math.ceil(len(index_array) / batch_size)
        else:
            index_array = list(range(len(data)))
            pickle.dump(index_array, open( "index_array", "wb" ))
            batch_num = math.ceil(len(index_array) / batch_size)

        if shuffle:
            np.random.shuffle(index_array)

        if start != 0:
            try:
                index_array = index_array[start:]
            except IndexError:
                print('start index out of index, ignore command')
            batch_num = math.ceil(len(index_array) / batch_size)
    else:
        index_array = list(range(len(data)))
        batch_num = math.ceil(len(index_array) / batch_size)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        if train:
            pickle.dump(indices, open( "current_index", "wb" ))
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
