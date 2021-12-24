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

predictions = []

x = 0
exact_match = 0
other = 0
scs = []
output = list()
for c, b in enumerate(batch):
    words = origin[c]['token']
    ss, se = origin[c]['subj_start'], origin[c]['subj_end']
    os, oe = origin[c]['obj_start'], origin[c]['obj_end']
    preds,sc = trainer.predict_with_saliency(b)
    if preds[0] != 0:
        print (id2label[preds[0]])
        saliency = []
        output = []
        i = 0
        print (sc)
        for j, t in enumerate(words):
            if j == ss or j == os:
                i += 1
            if j>=ss and j<=se:
                assert sc[i-1] == 0
                saliency.append(sc[i-1])
                output.append(colored(t, "blue"))
            elif j>=os and j<=oe:
                assert sc[i-1] == 0
                saliency.append(sc[i-1])
                output.append(colored(t, "yellow"))
            else:
                output.append(t)
                t = convert_token(t)
                sub_len = len(tokenizer.tokenize(t))
                saliency.append(sc[i: i+sub_len].mean())
                i += sub_len
        top3 = np.array(saliency).argsort()[-3:].tolist()
        output = [w if i not in top3 else colored(w, 'red') for i, w in enumerate(output)]
        print (" ".join(output))
    predictions += preds
    batch_size = len(preds)

        

# with open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
#     f.write(json.dumps(output))
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")
