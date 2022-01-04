"""
Run evaluation with saved models.
"""
import random
import argparse
import torch

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert import tokenization
from transformers import BertTokenizer

import json
from termcolor import colored

import numpy as np

import statistics
from lime.lime_text import LimeTextExplainer

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

def preprocess(filename, tokenizer):
    with open(filename) as infile:
        data = json.load(infile)
    output_tokens = list()
    labels = list()
    # random.shuffle(data)
    for c, d in enumerate(data):
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
            elif i>=os and i<=oe:
                pass
            else:
                t = convert_token(t)
                for j, sub_token in enumerate(tokenizer.tokenize(t)):
                    words.append(sub_token)
        words = ['[CLS]'] + words + ['[SEP]']
        output_tokens.append(words)
        relation = d['relation']
        labels.append(relation)
    return output_tokens, labels

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--device', type=int, default=0, help='Word embedding dimension.')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = tokenization.BertTokenizer.from_pretrained('spanbert-large-cased')
# vocab_file = tokenizer.save_vocabulary("saved_vocab/")
# tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
print (len(tokenizer.vocab))
# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)

x_test, y_test = preprocess(data_file, tokenizer)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    tagging = f.readlines()

def predict(texts):
    texts = [[x if x!='' else '[MASK]' for x in t.split(' ')] for t in texts]
    print ([tokenizer.convert_tokens_to_ids(t) for t in texts])
    tokens = np.array([tokenizer.convert_tokens_to_ids(t) for t in texts]).astype(int)
    scores = trainer.predict_proba(tokens.reshape(1, -1, 1))
    return scores

explainer = LimeTextExplainer(class_names=id2label, split_expression=' ')
predictions = list()
for i, t in enumerate(x_test):
    text = ' '.join(t)
    assert len(t) == len(text.split(' '))
    prob = predict([text])
    pred = np.argmax(prob, axis=1).tolist()[0]
    predictions.append(id2label[pred])
    l = label2id[y_test[i]]
    exp = explainer.explain_instance(text, predict, num_features=len(t), num_samples=2000, labels=[pred, l])

p, r, f1 = scorer.score(y_test, predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
