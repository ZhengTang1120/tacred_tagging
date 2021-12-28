import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import argparse
from tqdm import tqdm
import torch

from dataloader import DataLoader, convert_token
from trainer import BERTtrainer, unpack_batch
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert.tokenization import BertTokenizer

import json

from termcolor import colored

import numpy as np

import statistics

from tensorflow.python.keras.losses import binary_crossentropy
from cxplain import RNNModelBuilder, WordDropMasking, CXPlain
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

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
        tokens = tokenizer.convert_tokens_to_ids(words)
        words = [[0] if i>=len(tokens) else [tokens[i]] for i in range(128)]
        output_tokens.append(words)
        labels.append(constant.LABEL_TO_ID[d['relation']])
    return np.array(output_tokens).astype(int), np.array(labels).astype(int)

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

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

# load data
train_file = opt['data_dir'] + '/train.json'
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)

x_train, y_train = preprocess(train_file, tokenizer)
x_test, y_test = preprocess(data_file, tokenizer)
print (x_train.shape)
helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    tagging = f.readlines()

class EXModel:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, x):
        return self.model.predict_proba(x)

explained_model = EXModel(trainer)


model_builder = RNNModelBuilder(embedding_size=1024, with_embedding=True,
                                num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
                                batch_size=4, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
masking_operation = WordDropMasking()
loss = binary_crossentropy


explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

explainer.fit(x_train, y_train)

attributions = explainer.explain(x_test)


np.random.seed(909)
selected_index = np.random.randint(len(x_test))
selected_sample = x_test[selected_index]
importances = attributions[selected_index]

print (importances)