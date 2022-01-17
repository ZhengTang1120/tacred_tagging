"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert import tokenization
from transformers import BertTokenizer

import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--device', type=int, default=0, help='device')

parser.add_argument('--rationale', type=str, default='output_lime_132_test_best_model_6.json', help='rationale')

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
opt['rationale'] = args.rationale
trainer = BERTtrainer(opt)
trainer.load(model_file)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, 1))
opt['do_rationale'] = False
batch = DataLoader(data_file, 1, opt, tokenizer, True)
opt['do_rationale'] = True
batch_r = DataLoader(data_file, 1, opt, tokenizer, True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []

x = 0
exact_match = 0
other = 0
log_odds = list()
for c, b in enumerate(batch):
    preds, _, probs = trainer.predict(b, id2label, tokenizer)
    _, _, probs_r = trainer.predict(batch_r[c], id2label, tokenizer)
    print (preds.shape, probs.shape)
    p1 = np.take_along_axis(probs, preds.reshape(-1, 1),1)
    p2 = np.take_along_axis(probs_r, preds.reshape(-1, 1),1)
    for i, p in enumerate(preds):
        if p!=0:
            log_odd = p1[i]/(1.0-p1[i]) - p2[i]/(1.0-p2[i])
            log_odds.append(log_odd)
    predictions += preds.tolist()
    batch_size = len(preds.tolist())
output = list()
for i, p in enumerate(predictions):
        predictions[i] = id2label[p]

# with open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
#     f.write(json.dumps(output))
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
print (sum(log_odds)/len(log_odds))
print("Evaluation ended.")
