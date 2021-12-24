"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from dataloader import DataLoader, convert_token
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert.tokenization import BertTokenizer

import json

from termcolor import colored

import numpy as np

import statistics

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
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
origin = json.load(open(data_file))
print("Loading data from {} with batch size {}...".format(data_file, 1))
batch = DataLoader(data_file, 1, opt, tokenizer, True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    tagging = f.readlines()
predictions = []

x = 0
exact_match = 0
other = 0
scs = []
output = list()
tagging_scores = list()
for c, b in enumerate(batch):
    _, tagged = tagging[c].split('\t')
    tagged = eval(tagged)
    words = origin[c]['token']
    ss, se = origin[c]['subj_start'], origin[c]['subj_end']
    os, oe = origin[c]['obj_start'], origin[c]['obj_end']
    preds,sc = trainer.predict_with_saliency(b)
    output.append({'gold_label':batch.gold()[c], 'predicted_label':id2label[predictions[c]], 'predicted_tags':[], 'gold_tags':[]})
    if preds[0] != 0:
        saliency = []
        tokens = []
        i = 0
        for j, t in enumerate(words):
            if j == ss or j == os:
                i += 1
            if j>=ss and j<=se:
                assert sc[i-1] == 0
                saliency.append(sc[i-1])
                tokens.append(colored(t, "blue"))
            elif j>=os and j<=oe:
                assert sc[i-1] == 0
                saliency.append(sc[i-1])
                tokens.append(colored(t, "yellow"))
            else:
                tokens.append(t)
                t = convert_token(t)
                sub_len = len(tokenizer.tokenize(t))
                saliency.append(sc[i: i+sub_len].mean())
                i += sub_len
        top3 = np.array(saliency).argsort()[-3:].tolist()
        output[-1]["predicted_tags"] = top3
        tokens = [w if i not in top3 else colored(w, 'red') for i, w in enumerate(tokens)]
        if len(tagged)>0:
            output[-1]['gold_tags'] = tagged
            print (output[-1]['gold_label'], output[-1]['predicted_label'])
            print (" ".join(tokens))
            print (" ".join([w if i not in tagged else colored(w, 'red') for i, w in enumerate(words)]))
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
    predictions += preds
    batch_size = len(preds)

        

tr, tp, tf = zip(*tagging_scores)

print("{} set rationale result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))
with open("output_saliency_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
    f.write(json.dumps(output))
