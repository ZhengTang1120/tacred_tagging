import random
import argparse
import torch
from trainer import BERTtrainer
from dataloader import DataLoader
from utils import torch_utils, scorer, constant, helper
import json
from transformers import BertTokenizer
from termcolor import colored

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
        words = words[:128]
        processed.append(words)
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

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
for c, b in enumerate(batch):
    preds,_ = trainer.predict(b, id2label, tokenizer)
    predictions += preds
    batch_size = len(preds)

data = preprocess(data_file, tokenizer)

for c, words in enumerate(data):
    l = None
    rationale = list()
    while l!=predictions[c]:
        candidates = list()
        for i in range(len(words)):
            if i not in rationale:
                cand_r = rationale+[i]
                cand_r.sort()
                print (cand_r)
                ids = tokenizer.convert_tokens_to_ids(['[CLS]']+[words[j] for j in cand_r]+['[SEP]'])
                mask = [1] * len(ids)
                segment_ids = [0] * len(ids)
                candidates.append((ids, mask, segment_ids))
        candidates = list(zip(*candidates))
        print (candidates)
        with torch.cuda.device(args.device):
            inputs = [torch.LongTensor(c).cuda() for c in candidates]
        print ([x.size() for x in inputs])
        b, l = trainer.predict_cand(inputs, predictions[c])
        rationale.append(b)
    if predictions[c] != 0:
        print (id2label[predictions[c]])
        print (" ".join([w if i not in rationale else colored(w, 'red') for i, w in enumerate(words)]))





