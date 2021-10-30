"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import string

from utils import constant, helper
from collections import defaultdict
from statistics import mean


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, tokenizer):
        self.batch_size = batch_size
        self.opt = opt
        self.label2id = constant.LABEL_TO_ID
        self.tokenizer = tokenizer

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, opt)
        data = sorted(data, key=lambda f: len(f[0]))
        
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.words = [d[-1] for d in data]
        self.num_examples = len(data)
        
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(self.data), filename))

    def preprocess(self, data, opt):


        """ Preprocess the data and convert to ids. """
        processed = []
        processed_rule = []
        for c, d in enumerate(data):
            tokens = list()
            words  = list()
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            for i, t in enumerate(d['token']):
                if i == ss:
                    words.append("[unused%d]"%constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+d['subj_type']+']'])
                if i == os:
                    words.append("[unused%d]"%constant.ENTITY_TOKEN_TO_ID['[OBJ-'+d['obj_type']+']'])
                if i>=ss and i<=se:
                    pass
                elif i>=os and i<=oe:
                    pass
                else:
                    t = convert_token(t)
                    for sub_token in self.tokenizer.tokenize(t):
                        words.append(sub_token)
            words = ['[CLS]'] + words + ['[SEP]']
            relation = self.label2id[d['relation']]
            tokens = [self.tokenizer.convert_tokens_to_ids(w) for w in words]
            if len(tokens) > 128:
                tokens = tokens[:128]
            mask = [1] * len(tokens)
            segment_ids = [0] * len(tokens)
            processed += [(tokens, mask, segment_ids, relation, words)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        

        # sort all fields by lens
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        # word dropout
        words = batch[0]
        print (words)
        mask = batch[1]
        segment_ids = batch[2]
        # convert to tensors
        words = get_long_tensor(words, batch_size)
        mask = get_long_tensor(mask, batch_size)
        segment_ids = get_long_tensor(segment_ids, batch_size)

        rels = torch.LongTensor(batch[-2])#

        return (words, mask, segment_ids, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i,:len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token
