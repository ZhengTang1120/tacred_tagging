from bert import BERTencoder, BertForMaskedLM
import random
import argparse
import torch
from dataloader import DataLoader
from trainer import unpack_batch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import torch_utils, scorer, constant, helper
from trainer import BERTtrainer

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--device', type=int, default=0, help='Word embedding dimension.')

parser.add_argument('--loaded', type=bool, default=True, help='Word embedding dimension.')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('spanbert-large-cased')


if args.loaded:
    # load opt
    model_file = args.model_dir + '/' + args.model
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    opt['device'] = args.device
    trainer = BERTtrainer(opt)
    trainer.load(model_file)
    lm = BertForMaskedLM(trainer.encoder.model)
else:
    encoder = BERTencoder()
    lm = BertForMaskedLM(encoder.model)

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, tokenizer, True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

nlls = []
for c, b in enumerate(batch):
    inputs, labels, has_tag = unpack_batch(b, opt['cuda'], opt['device'])
    words = inputs[0]
    mask = inputs[1]
    segment_ids = inputs[2]
    with torch.no_grad():
        neg_log_likelihood = lm(words, mask, segment_ids, words)
    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print (ppl)








