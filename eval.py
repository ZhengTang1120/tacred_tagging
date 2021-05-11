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

from transformers import BertTokenizer

import json

from lime import lime_text
from lime.lime_text import LimeTextExplainer

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
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
special_tokens_dict = {'additional_special_tokens': constant.ENTITY_TOKENS}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.encoder.model.resize_token_embeddings(len(tokenizer)) 
trainer.load(model_file)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = [-1 for k in label2id]
for k,v in label2id.items():
    id2label[v] = k
explainer = LimeTextExplainer(class_names=id2label,split_expression='=SEP=')
predictions = []
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def predict(text):
    text = [t.split('=SEP=') for t in text]
    tokens = [tokenizer.convert_tokens_to_ids(t) for t in text]
    probs = None
    for batch in chunks(tokens, 40):
        probs = trainer.predict_text(batch) if probs is None else np.concatenate((probs, trainer.predict_text(batch)), axis=0)
    return probs
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    tagged_ids = f.readlines()
for i, raw in enumerate(batch.words):
    text = ['=SEP='.join(raw)]
    probs = predict(text)
    ol, tagged = tagged_ids[i].split('\t')
    tagged = eval(tagged)
    if tagged and batch.gold()[i] != 'no_relation':
        l = label2id[batch.gold()[i]]
        exp = explainer.explain_instance(text[0], predict, num_features=len(raw), num_samples=2000, labels=[l])
        print (batch.gold()[i], tagged, exp.as_map()[l], exp.as_list(label=l))
        print (raw)
        exp.save_to_file('lime_test.html')
        exit()
        # r = sum([1 if t[j]==rules[i][j] else 0 for j in range(len(t)) if rules[i][j]!=0])/sum(rules[i])
        # pr = sum([1 if t[j]==rules[i][j] else 0 for j in range(len(t)) if rules[i][j]!=0])/sum(t) if sum(t)!=0 else 0
    pred = np.argmax(probs, axis=1).tolist()
    predictions += [id2label[pred[0]]]

p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))


