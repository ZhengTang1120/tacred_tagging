import random
import argparse
import torch
from trainer import BERTtrainer
from dataloader import DataLoader
from utils import torch_utils, scorer, constant, helper
import json
from transformers import BertTokenizer
from termcolor import colored

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max([len(x) for x in tokens_list])
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i,:len(s)] = torch.LongTensor(s)
    return tokens

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
    processed = list()
    for c, d in enumerate(data):
        tokens = list()
        words  = list()
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        sub_token_len = 0
        for i, t in enumerate(d['token']):
            if sub_token_len >= 128:
                break
            if i == ss or i == os:
                sub_token_len += 1
            if i>=ss and i<=se:
                words.append(colored(t, 'blue'))
            elif i>=os and i<=oe:
                words.append(colored(t, 'yellow'))
            else:
                t = convert_token(t)
                words.append(t)
                sub_token_len += len(tokenizer.tokenize(t))
        processed.append((words, ss, se, os, oe, d['subj_type'], d['obj_type']))
    return processed

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

tokenizer = BertTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer, True)

with open(opt['data_dir'] + '/tagging_{}.txt'.format(args.dataset)) as f:
    tagging = f.readlines()

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
for c, b in enumerate(batch):
    preds,_ = trainer.predict(b, id2label, tokenizer)
    predictions += preds
    batch_size = len(preds)

data = preprocess(data_file, tokenizer)
tagging_scores = list()
for c, d in enumerate(data):
    words, ss, se, os, oe, subj, obj = d
    _, tagged = tagging[c].split('\t')
    tagged = eval(tagged)
    if predictions[c] != 0:
        l = None
        rationale = [ss, os]
        while l!=predictions[c]:
            candidates = list()
            cr = list()

            for i in range(len(words)):
                if i not in rationale and i not in range(ss, se+1) and i not in range(os, oe+1):
                    cr.append(i)
                    cand_r = rationale+[i]
                    cand_r.sort()
                    tokens = []
                    for j in cand_r:
                        if j == ss or j == os:
                            tokens.append("[unused%d]"%(constant.ENTITY_TOKEN_TO_ID['[OBJ-'+obj+']']+1))
                        else:
                            tokens += tokenizer.tokenize(words[j])
                    ids = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                    mask = [1] * len(ids)
                    segment_ids = [0] * len(ids)
                    candidates.append((ids, mask, segment_ids))
            if len(candidates)!=0:
                candidates = list(zip(*candidates))
                with torch.cuda.device(args.device):
                    inputs = [get_long_tensor(c, len(c)).cuda() for c in candidates]
                b, l = trainer.predict_cand(inputs, predictions[c])
                rationale.append(cr[b])
            else:
                rationale = [ss, os]
        print (id2label[predictions[c]])
        rationale.remove(ss)
        rationale.remove(os)
        print (" ".join([w if i not in rationale else colored(w, 'red') for i, w in enumerate(words)]))
        if len(tagged)>0:
            correct = 0
            pred = 0
            for j, t in enumerate(words):
                if j in rationale and j in tagged:
                    pred += 1
                    if j in tagged:
                        correct += 1
            if pred > 0:
                r = correct / pred
            else:
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

tr, tp, tf = zip(*tagging_scores)

print("{} set rationale result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,math.mean(tr),math.mean(tp),math.mean(tf)))





