import random
import argparse
import torch
from trainer import BERTtrainer
from dataloader import DataLoader
from utils import torch_utils, scorer, constant, helper
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from termcolor import colored

import statistics

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max([len(x) for x in tokens_list])
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i,:len(s)] = torch.LongTensor(s)
    return tokens

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

def preprocess(data, tokenizer):
    
    processed = list()
    for c, d in enumerate(data):
        tokens = list()
        words  = list()
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        sub_token_len = 0
        for i, t in enumerate(d['token']):
            if sub_token_len >= 128:
                break
            if i == ss or i == os:
                sub_token_len += 1
            if i>=ss and i<=se:
                words.append(colored(t, 'blue'))
            elif i>=os and i<=oe:
                words.append(colored(t, 'yellow'))
            else:
                t = convert_token(t)
                words.append(t)
                sub_token_len += len(tokenizer.tokenize(t))
        processed.append((words, ss, se, os, oe, d['subj_type'], d['obj_type']))
    return processed

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', type=int, default=0, help='Word embedding dimension.')
parser.add_argument('--dataset', type=str, default='train', help="Evaluate on dev or test.")

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')
opt = vars(args)
if args.dataset == "train":
    train_file = opt['data_dir'] + '/train.json'
    dev_file = opt['data_dir'] + '/dev.json'

    trainer = BERTtrainer(opt)

    with open(train_file) as infile:
        tdata = json.load(infile)
    train_data = preprocess(data, tokenizer)

    with open(dev_file) as infile:
        ddata = json.load(infile)
    dev_data = preprocess(data, tokenizer)

    for epoch in range(10):
        f1 = 0
        for c, d in enumerate(train_data):
            words, ss, se, os, oe, subj, obj = d

            rationale = [ss, os]
            score = 0
            probs = None
            while True:
                candidates = list()
                cr = list()
                for i in range(len(words)):
                    if i not in rationale and i not in range(ss, se+1) and i not in range(os, oe+1):
                        cr.append(i)
                        cand_r = rationale+[i]
                        cand_r.sort()
                        tokens = []
                        for j in cand_r:
                            if j == ss:
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+obj+']']+1))
                            if j == os:
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1))
                            else:
                                tokens += tokenizer.tokenize(words[j])
                        ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                        mask = [1] * len(ids)
                        segment_ids = [0] * len(ids)
                        candidates.append((ids, mask, segment_ids))
                if len(candidates)!=0:
                    candidates = list(zip(*candidates))
                    with torch.cuda.device(args.device):
                        inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                    b, p = trainer.update_cand(inputs, predictions[c])
                    if score < p[r]:
                        score = p[r]
                        rationale.append(cr[b])
                    else:
                        probs = p
                        break
                else:
                    tokens = ["[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+obj+']']+1), "[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1)]
                    ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                    mask = [1] * len(ids)
                    segment_ids = [0] * len(ids)
                    candidates.append((ids, mask, segment_ids))
                    candidates = list(zip(*candidates))
                    with torch.cuda.device(args.device):
                        inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                    b, _, p = trainer.update_cand(inputs, predictions[c])
                    probs = p
                    break
            loss = trainer.criterion(probs, r)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        predictions = []
        golds = []
        for c, d in enumerate(dev_data):
            words, ss, se, os, oe, subj, obj = d
            golds.append(ddata[c]['relation'])
            rationale = [ss, os]
            score = 0
            probs = None
            while True:
                candidates = list()
                cr = list()
                for i in range(len(words)):
                    if i not in rationale and i not in range(ss, se+1) and i not in range(os, oe+1):
                        cr.append(i)
                        cand_r = rationale+[i]
                        cand_r.sort()
                        tokens = []
                        for j in cand_r:
                            if j == ss:
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+obj+']']+1))
                            if j == os:
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1))
                            else:
                                tokens += tokenizer.tokenize(words[j])
                        ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                        mask = [1] * len(ids)
                        segment_ids = [0] * len(ids)
                        candidates.append((ids, mask, segment_ids))
                if len(candidates)!=0:
                    candidates = list(zip(*candidates))
                    with torch.cuda.device(args.device):
                        inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                    b, s, p = trainer.predict_cand2(inputs, score)
                    if b != -1:
                        score = s
                        rationale.append(cr[b])
                    else:
                        predictions.append(id2label[p])
                        break
                else:
                    tokens = ["[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+obj+']']+1), "[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1)]
                    ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                    mask = [1] * len(ids)
                    segment_ids = [0] * len(ids)
                    candidates.append((ids, mask, segment_ids))
                    candidates = list(zip(*candidates))
                    with torch.cuda.device(args.device):
                        inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                    b, s, p = trainer.predict_cand2(inputs, score)
                    predictions.append(id2label[p])
                    break
        dev_p, dev_r, dev_f1 = scorer.score(golds, predictions)
        if dev_f1 > f1:    
            model_file = 'saved_models/250/best_model.pt'
            trainer.save(model_file)
            f1 = dev_f1

    



