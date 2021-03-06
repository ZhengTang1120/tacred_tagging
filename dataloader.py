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
    def __init__(self, filename, batch_size, opt, tokenizer, tagging, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.tokenizer = tokenizer
        self.tagging = tagging

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-4]] for d in data]
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
        with open(self.tagging) as f:
            tagged_ids = f.readlines()
        for c, d in enumerate(data):
            tokens = list(d['token'])
            words  = list(d['token'])
            for i in range(len(words)):
                if words[i] == '-LRB-':
                    words[i] = '('
                if words[i] == '-RRB-':
                    words[i] = ')'
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['[SUBJ-'+d['subj_type']+']'] * (se-ss+1)
            tokens[os:oe+1] = ['[OBJ-'+d['obj_type']+']'] * (oe-os+1)
            ol, tagged = tagged_ids[c].split('\t')
            tagged = eval(tagged)
            ner = d['stanford_ner']
            if len(tagged)!=0 and d['relation'] != 'no_relation' and d['relation'] == ol:
                for i in range(len(tagged)):
                    tagged[i] += 1
                has_tag = True
            else:
                tagged = []
                pattern = ''
                has_tag = False
            tokens = ['[CLS]'] + tokens
            words = ['[CLS]'] + words
            ner = ['CLS'] + ner
            relation = self.label2id[d['relation']]
            if has_tag:
                tagging = [0 if i not in tagged else 1 for i in range(len(tokens))]
            else:
                tagging = [0 for i in range(len(tokens))]
            l = len(tokens)
            for i in range(l):
                if tokens[i] == '-LRB-':
                    tokens[i] = '('
                if tokens[i] == '-RRB-':
                    tokens[i] = ')'
            if ss<os:
                entity_positions = get_positions2(ss+1, se+1, os+1, oe+1, l)
            else:
                entity_positions = get_positions2(os+1, oe+1, ss+1, se+1, l)
            subj_positions = get_positions(ss+1, se+1, l)
            obj_positions = get_positions(os+1, oe+1, l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            processed += [(tokens, entity_positions, subj_positions, obj_positions, subj_type, obj_type, relation, tagging, has_tag, words)]
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
        

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        # word dropout
        words = batch[0]
        # convert to tensors
        words = get_long_tensor(words, batch_size)
        # words = self.tokenizer(batch[0], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
        entity_positions = get_long_tensor(batch[1], batch_size)
        subj_positions = get_long_tensor(batch[2], batch_size)
        obj_positions = get_long_tensor(batch[3], batch_size)
        subj_type = get_long_tensor(batch[4], batch_size)
        obj_type = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])#

        rule = get_long_tensor(batch[7], batch_size)
        masks = torch.eq(rule, 0)
        return (words, masks, entity_positions, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx, rule, batch[-2])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [constant.SOS_ID] + [vocab[t] if t in vocab else constant.UNK_ID for t in tokens] + [constant.EOS_ID]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [1000]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_positions2(s1, e1, s2, e2, length):
    """ Get subj&obj position sequence. """
    return [1] + [3] * (s1 - 1) + \
            [2] * (e1 - s1 + 1) + \
            [4] * (s2 - e1 - 1) +\
            [2] * (e2 - s2 + 1) + \
            [5] * (length - e2 - 1)

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

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]