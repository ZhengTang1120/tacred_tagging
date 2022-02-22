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

from pytorch_pretrained_bert.tokenization import BertTokenizer

import json

from termcolor import colored

import statistics

def check(tags, ids):
    for i in ids:
        if i<len(tags) and tags[i] == 1:
            return True
    return False

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
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer, True)


with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset.replace("_tacred",""))) as f:
    tagging = f.readlines()

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []

x = 0
exact_match = 0
other = 0
tags = []
tagging_max = []
words = []
for c, b in enumerate(batch):
    preds,t,_,tm, w = trainer.predict(b, id2label, tokenizer)
    predictions += preds
    tags += t
    words += w
    tagging_max += tm
    batch_size = len(preds)
output = list()
tagging_scores = []
output = list()
pred_output = open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.txt')), 'w')
for i, p in enumerate(predictions):
    _, tagged = tagging[i].split('\t')
    tagged = eval(tagged)
    predictions[i] = id2label[p]
    pred_output.write(id2label[p]+'\n')
    output.append({'gold_label':batch.gold()[i], 'predicted_label':id2label[p], 'predicted_tags':[], 'gold_tags':[]})
    if i in [1442, 1722, 2111, 2409]:
        print (words[i])
        print (tagging_max[i])
        print (tags[i])
        print (tags[i].index(1))
        print (batch.words[i])
        print (id2label[p], batch.gold()[i], tagged, [j for j, t in enumerate(batch.words[i]) if check(tags[i], t[1])])

    if p!=0:
        output[-1]["predicted_tags"] = [j for j, t in enumerate(batch.words[i]) if check(tags[i], t[1])]
        if len(tagged)>0:
            output[-1]['gold_tags'] = tagged
            correct = 0
            pred = 0
            for j, t in enumerate(batch.words[i]):
                if check(tags[i], t[1]):
                    pred += 1
                    if j in tagged:
                        correct += 1
            if pred > 0:
                r = correct / pred
            else:
                print (tags[i])
                r = 0
            if len(tagged) > 0:
                p = correct / len(tagged)
            else:
                p = 0
            try:
                f1 = 2.0 * p * r / (p + r)
            except ZeroDivisionError:
                f1 = 0
            tagging_scores.append((r, p, f1))
pred_output.close()
with open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
    f.write(json.dumps(output))

tr, tp, tf = zip(*tagging_scores)
# with open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
#     f.write(json.dumps(output))
p, r, f1, ba = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
print("{} set tagging  result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))
print("Evaluation ended.")
