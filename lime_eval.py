"""
Run evaluation with saved models.
"""
import random
import argparse
import torch

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from pytorch_pretrained_bert import tokenization
from transformers import BertTokenizer

import json
from termcolor import colored

import numpy as np

import statistics
from lime.lime_text import LimeTextExplainer

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
        # if len(words) > 128:
        #         words = words[:128]
        output_tokens.append(words)
        relation = d['relation']
        labels.append(relation)
    return output_tokens, labels

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

tokenizer = tokenization.BertTokenizer.from_pretrained('spanbert-large-cased')
# vocab_file = tokenizer.save_vocabulary("saved_vocab/")
# tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
print (len(tokenizer.vocab))
# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)

x_test, y_test = preprocess(data_file, tokenizer)
origin = json.load(open(data_file))
helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset.replace("_tacred",""))) as f:
    tagging = f.readlines()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def predict(texts):
    texts = [[x if x!='' else '[MASK]' for x in t.split(' ')] for t in texts]
    tokens = [tokenizer.convert_tokens_to_ids(t) for t in texts]
    probs = None
    for batch in chunks(tokens, 32):
        batch = np.array(batch).astype(int).reshape(len(batch), -1, 1)
        probs = trainer.predict_proba(batch) if probs is None else np.concatenate((probs, trainer.predict_proba(batch)), axis=0)
    return probs
output = list()
explainer = LimeTextExplainer(class_names=id2label, split_expression=' ')
predictions = list()
tagging_scores = list()
for i, t in enumerate(x_test):
    _, tagged = tagging[i].split('\t')
    tagged = eval(tagged)
    words = origin[i]['token']
    ss, se = origin[i]['subj_start'], origin[i]['subj_end']
    os, oe = origin[i]['obj_start'], origin[i]['obj_end']
    text = ' '.join(t)
    assert len(t) == len(text.split(' '))
    prob = predict([text])
    pred = np.argmax(prob, axis=1).tolist()[0]
    predictions.append(id2label[pred])
    l = label2id[y_test[i]]
    output.append({'gold_label':y_test[i], 'predicted_label':id2label[pred], 'predicted_tags':[], 'gold_tags':[]})
    if id2label[pred] != 'no_relation':
        exp = explainer.explain_instance(text, predict, num_features=len(t), num_samples=2000, labels=[pred, l])
        importance = {x[0]:x[1] for x in exp.as_list(label=pred)}
        saliency = []
        tokens = []
        for j, x in enumerate(words):
            if j< 128:
                if j>=ss and j<=se:
                    saliency.append(0)
                    tokens.append(colored(x, "blue"))
                elif j>=os and j<=oe:
                    saliency.append(0)
                    tokens.append(colored(x, "yellow"))
                else:
                    tokens.append(convert_token(x))
                    saliency.append(statistics.mean([importance[xx] for xx in tokenizer.tokenize(convert_token(x))]))
            else:
                tokens.append(x)
                saliency.append(0)
        top3 = np.array(saliency).argsort()[-3:].tolist()
        output[-1]["predicted_tags"] = saliency
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
with open("output_lime_{}_{}_{}".format(args.model_dir.split('/')[-1], args.dataset, args.model.replace('.pt', '.json')), 'w') as f:
    f.write(json.dumps(output))

p, r, f1 = scorer.score(y_test, predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
