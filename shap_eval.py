import random
import argparse
import torch
from dataloader import DataLoader, convert_token
from trainer import BERTtrainer, unpack_batch
from utils import torch_utils, scorer, constant, helper
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from pytorch_pretrained_bert import tokenization
from transformers import BertTokenizer
import json
from termcolor import colored
import numpy as np
import scipy as sp
import statistics
import shap

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
        labels.append(d['relation'])
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
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

tokenizer = tokenization.BertTokenizer.from_pretrained('spanbert-large-cased')
vocab_file = tokenizer.save_vocabulary("saved_vocab/")
tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)

x_test, y_test = preprocess(data_file, tokenizer)

def f(x):
    print (x)
    tv = np.array([tokenizer.encode(v, pad_to_max_length=True, max_length=128,truncation=True) for v in x]).astype(int)
    scores = trainer.predict_proba(tv.reshape(1, -1, 1))
    val = sp.special.logit(scores)
    print (val)
    return val

explainer = shap.Explainer(f, tokenizer, output_names=sorted(constant.LABEL_TO_ID, key=constant.LABEL_TO_ID.get))

shap_values = explainer(x_test[:3])
print (shap_values)