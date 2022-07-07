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

origin = json.load(open(data_file))
# hide_relations = ["per:employee_of", "per:age", "org:city_of_headquarters", "org:country_of_headquarters", "org:stateorprovince_of_headquarters", "per:origin"]
tagging = []
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    # tagging = f.readlines()
    for i, line in enumerate(f):
        tagging.append(line)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []

x = 0
exact_match = 0
other = 0
for c, b in tqdm(enumerate(batch)):
    preds,_ = trainer.predict(b, id2label, tokenizer)
    predictions += preds
    batch_size = len(preds)
output = list()
tagging_scores = []
output = list()
pred_output = open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.txt')), 'w')
for i, p in enumerate(predictions):
    tokens = []
    tokens2 = []

    _, tagged = tagging[i].split('\t')
    tagged = eval(tagged)
    
    predictions[i] = id2label[p]
    pred_output.write(id2label[p]+'\n')
pred_output.close()
p, r, f1, ba = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p*100,r*100,f1*100))
