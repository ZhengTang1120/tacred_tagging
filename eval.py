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
model_file = args.model_dir + '_0/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device


# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer,  opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset), evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
tags = []
goldt = []
inputs = []
subjs = []
objs = []

chunks = np.array_split(np.array(range(len(batch))),5)

for x, ch in enumerate(chunks):
    model_file = args.model_dir + '_%d/'%x + args.model
    trainer = BERTtrainer(opt)
    trainer.encoder.model.resize_token_embeddings(len(tokenizer)) 
    trainer.load(model_file)
    for c, b in enumerate(batch):
        if c in ch:
            preds, ts, tagged, ids, s, o = trainer.predict(b, id2label, tokenizer)
            predictions += preds
            tags += ts
            subjs += s
            objs += o
            goldt += tagged
            batch_size = len(preds)
            for i in range(batch_size):
                inputs += [[tokenizer.convert_ids_to_tokens(j) for j in ids[i]]]
output = list()
for i, p in enumerate(predictions):
    predictions[i] = id2label[p]
    output.append({'gold_label':batch.gold()[i], 'predicted_label':id2label[p], 'raw_words':batch.words[i], 'predicted_tags':[], 'gold_tags':[], 'subj':[], 'obj':[]})
    output[-1]['subj'] = [subjs[i][j] for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]']
    output[-1]['obj'] = [objs[i][j] for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]']
    if sum(goldt[i])!=0:
        output[-1]['gold_tags'] = [goldt[i][j] for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]']
            # print (id2label[p], batch.gold()[i])
            # print ([(goldt[i][j], tags[i][j], batch.words[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
            # print ()
    if p!=0 and sum(tags[i])!=0:
        output[-1]['predicted_tags'] = [tags[i][j] for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]']
            # print (id2label[p], batch.gold()[i])
            # print ([(tags[i][j], batch.words[i][j])for j in range(len(inputs[i])) if inputs[i][j] != '[PAD]'])
            # print ()
# with open("output_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
#     f.write(json.dumps(output))
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

