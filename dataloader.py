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


class DataProcessor(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, opt, tokenizer, eval=False):
        self.opt = opt
        self.label2id = constant.LABEL_TO_ID
        self.tokenizer = tokenizer

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, opt)
        if not eval:
            data = sorted(data, key=lambda f: len(f[0]))
        
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.words = [d[-1] for d in data]
        self.num_examples = len(data)
        
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
                    words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+d['subj_type']+']']+1))
                if i == os:
                    words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+d['obj_type']+']']+1))
                if i>=ss and i<=se:
                    pass
                    # words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+d['subj_type']+']']+1))
                elif i>=os and i<=oe:
                    pass
                    # words.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+d['obj_type']+']']+1))
                else:
                    t = convert_token(t)
                    for sub_token in self.tokenizer.tokenize(t):
                        words.append(sub_token)
            words = ['[CLS]'] + words + ['[SEP]']
            relation = self.label2id[d['relation']]

            tokens = self.tokenizer.convert_tokens_to_ids(words)
            if len(tokens) > 128:
                tokens = tokens[:128]
            segment_ids = [0] * len(tokens)
            mask = [1] * len(tokens)
            padding = [0] * (128 - len(tokens))
            tokens += padding
            mask += padding
            segment_ids += padding
            assert len(tokens) == 128
            assert len(mask) == 128
            assert len(segment_ids) == 128
            processed += [(tokens, mask, segment_ids, relation, words)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

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
