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
        tokens = tokenizer.convert_tokens_to_ids(words)
        words = [[0] if i>=len(tokens) else [tokens[i]] for i in range(128)]
        output_tokens.append(words)
        relation = [1 if i == constant.LABEL_TO_ID[d['relation']] else 0 for i in range(len(constant.LABEL_TO_ID))]
        labels.append(relation)
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

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')
print (len(tokenizer.vocab))
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

origin = json.load(open(data_file))
# model_builder = RNNModelBuilder(embedding_size=len(tokenizer.vocab), with_embedding=True,
#                                 num_layers=2, num_units=32, activation="relu", p_dropout=0.2, verbose=0,
#                                 batch_size=32, learning_rate=0.001, num_epochs=2, early_stopping_patience=128)
# masking_operation = WordDropMasking()
# loss = binary_crossentropy


# explainer = CXPlain(explained_model, model_builder, masking_operation, loss)

# explainer.fit(x_train, y_train)

my_save_directory = "cxplain_new/"
# explainer.save(my_save_directory)
output = list()
explainer = CXPlain.load(my_save_directory)
attributions = explainer.explain(x_test)
preds = list()
golds = list()
tagging_scores = list()
for i, t in enumerate(x_test):
    _, tagged = tagging[i].split('\t')
    tagged = eval(tagged)
    words = origin[i]['token']
    ss, se = origin[i]['subj_start'], origin[i]['subj_end']
    os, oe = origin[i]['obj_start'], origin[i]['obj_end']
    prob = explained_model.predict_proba(t.reshape(1, -1, 1))
    pred = id2label[np.argmax(prob, axis=1).tolist()[0]]
    preds.append(pred)
    golds.append(id2label[np.argmax(y_test[i])])
    importance = attributions[i][1:-1]
    output.append({'gold_label':golds[-1], 'predicted_label':preds[-1], 'predicted_tags':[], 'gold_tags':[]})
    if preds[-1] != 'no_relation':
        saliency = []
        tokens = []
        c = 0
        for j, t in enumerate(words):
            if j == ss or j == os:
                c += 1
            if j>=ss and j<=se:
                saliency.append(0)
                tokens.append(colored(t, "blue"))
            elif j>=os and j<=oe:
                saliency.append(0)
                tokens.append(colored(t, "yellow"))
            else:
                tokens.append(t)
                t = convert_token(t)
                sub_len = len(tokenizer.tokenize(t))
                saliency.append(importance[c: c+sub_len].mean())
                c += sub_len
        top3 = np.array(saliency).argsort()[-3:].tolist()
        output[-1]["predicted_tags"] = top3
        tokens = [w if c not in top3 else colored(w, 'red') for c, w in enumerate(tokens)]
        print (" ".join(tokens))
        if len(tagged)>0:
            output[-1]['gold_tags'] = tagged
            # print (saliency)
            # print (output[-1]['gold_label'], output[-1]['predicted_label'])
            # print (" ".join(tokens))
            # print (" ".join([w if i not in tagged else colored(w, 'red') for i, w in enumerate(words)]))
            correct = 0
            pred = 0
            for j, t in enumerate(words):
                if j in top3 and j in tagged:
                    correct += 1
            r = correct / 3
            if len(tagged) > 0:
                p = correct / len(tagged)
            else:
                p = 0
            try:
                f1 = 2.0 * p * r / (p + r)
            except ZeroDivisionError:
                f1 = 0
            tagging_scores.append((r, p, f1))
            print (r, p, f1)
            print ()

tr, tp, tf = zip(*tagging_scores)

print("{} set rationale result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))
with open("output_cxplain_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
    f.write(json.dumps(output))

p, r, f1 = scorer.score(golds, preds, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))