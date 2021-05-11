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
# explainer = LimeTextExplainer(class_names=id2label)
predictions = []

def predict(text):
    tokens = tokenizer.convert_tokens_to_ids(text)
    probs = trainer.predict_text(tokens)
    return probs

for i, text in enumerate(batch.words):
    probs = predict(text)
    # exp = explainer.explain_instance(text, predict, num_features=6)
    l = label2id[batch.gold()[i]]
    # print ('Explanation for class %s' % id2label[l])
    # print ('\n'.join(map(str, exp.as_list(label=l))))
    # print ()
    print (probs)
    pred = np.argmax(probs, axis=1).tolist()
    print (pred, l)
    print ()
    predictions += pred
    

p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

