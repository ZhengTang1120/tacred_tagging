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
import time

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
        if ss <= len(words) or os <= len(words):
            processed.append((words, ss, se, os, oe, d['subj_type'], d['obj_type']))
    return processed

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', type=int, default=0, help='Word embedding dimension.')
parser.add_argument('--dataset', type=str, default='train', help="Evaluate on dev or test.")
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--warmup_prop', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for.')


args = parser.parse_args()
opt = vars(args)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)
id2label = dict([(v,k) for k,v in label2id.items()])
tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')

if args.dataset == "train":
    train_file = opt['data_dir'] + '/train.json'
    dev_file = opt['data_dir'] + '/dev.json'

    

    with open(train_file) as infile:
        tdata = json.load(infile)
    train_data = preprocess(tdata, tokenizer)

    with open(dev_file) as infile:
        ddata = json.load(infile)
    dev_data = preprocess(ddata, tokenizer)

    opt['steps'] = len(tdata) * 10

    trainer = BERTtrainer(opt)
    for epoch in range(10):
        f1 = 0
        start_time = time.time()
        random.shuffle(train_data)
        for c, d in enumerate(train_data):
            words, ss, se, os, oe, subj, obj = d
            r = label2id[tdata[c]['relation']]
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
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+subj+']']+1))
                            if j == os:
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1))
                            else:
                                tokens += tokenizer.tokenize(words[j])
                        ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                        mask = [1] * len(ids)
                        segment_ids = [0] * len(ids)
                        candidates.append((ids, mask, segment_ids))
                if len(candidates)!=0:
                    chunks = [candidates[x:x+32] for x in range(0, len(candidates), 32)]
                    probs = None
                    for candidates in chunks:
                        candidates = list(zip(*candidates))
                        with torch.cuda.device(args.device):
                            inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                        o = trainer.update_cand(inputs)
                        if probs is None:
                            probs = o
                        else:
                            probs = torch.cat([probs, o], dim = 0)
                    b = np.argmax(probs.data.cpu().numpy(), axis=0).tolist()[r]
                    p = probs[b]
                    if score < p[r]:
                        score = p[r]
                        rationale.append(cr[b])
                    else:
                        probs = p.unsqueeze(0)
                        break
                else:
                    tokens = ["[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+subj+']']+1), "[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1)]
                    ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                    mask = [1] * len(ids)
                    segment_ids = [0] * len(ids)
                    candidates.append((ids, mask, segment_ids))
                    candidates = list(zip(*candidates))
                    with torch.cuda.device(args.device):
                        inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                    probs = trainer.update_cand(inputs)
                    b = np.argmax(probs.data.cpu().numpy(), axis=0).tolist()[r]
                    probs = probs[b].unsqueeze(0)
                    break
            label = torch.LongTensor([r]).cuda()
            loss = trainer.criterion(probs, label)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            print (loss.item())

        predictions = []
        golds = []
        for c, d in enumerate(dev_data):
            words, ss, se, os, oe, subj, obj = d
            golds.append(ddata[c]['relation'])
            rationale = [ss, os]
            score = 0
            pred = None
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
                                tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+subj+']']+1))
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
                        pred = p
                        rationale.append(cr[b])
                    else:
                        predictions.append(id2label[pred])
                        break
                else:
                    tokens = ["[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[SUBJ-'+subj+']']+1), "[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1)]
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
        duration = time.time() - start_time
        print (duration)

    



