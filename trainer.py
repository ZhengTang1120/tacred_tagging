"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder, BERTclassifier
from utils import constant, torch_utils

from transformers import AdamW

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        device = self.opt['device']
        self.opt = checkpoint['config']
        self.opt['device'] = device

    def save(self, filename, epoch):
        params = {
                'classifier': self.classifier.state_dict(),
                'encoder': self.encoder.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda, device):
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[0].to('cuda')]
            labels = Variable(batch[1].cuda())
            rules  = Variable(batch[3]).cuda()
    else:
        inputs = [Variable(batch[0])]
        labels = Variable(batch[1])
        rules  = Variable(batch[3])
    tokens = batch[0]
    return inputs, labels, tokens, rules, batch[-1]

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]
        if opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                self.encoder.cuda()
                self.classifier.cuda()
                self.criterion.cuda()
        self.optimizer = AdamW(
            self.parameters,
            lr=opt['lr'],
        )
    
    def update(self, batch, epoch):
        inputs, labels, tokens, _, _ = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        loss = 0
        o, b_out = self.encoder(inputs)
        h = o.pooler_output
        logits = self.classifier(h)
        loss += self.criterion(logits, labels)
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
        else:
            loss_val = 0
        return loss_val

    def predict(self, batch, id2label, tokenizer, unsort=True):
        inputs, labels, tokens, rules, tagged = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        tokens = tokens.data.cpu().numpy().tolist()
        orig_idx = batch[2]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        o, b_out = self.encoder(inputs)
        a = o.attentions
        a = a[-1].permute(2,0,1,3)[0].data.cpu().numpy().tolist()
        h = o.pooler_output
        logits = self.classifier(h)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        for i, p in enumerate(predictions):
            for k in range(16):
                print (a[k])
                print (sum(a[k]))
                print (o.attentions[-1][i].permute(2,0,1)[0][k])
                print (sum(o.attentions[-1][i].permute(2,0,1)[0][k]))
        #     if sum(rules[i])!=0 and tagged[i]:
        #             prs = []
        #             for k in range(len(a[i])):
        #                 top_attn = a[i][k][0].argsort()[:sum(rules[i])+1]
        #                 r = sum([1 if j in top_attn else 0 for j in range(len(rules[i])) if rules[i][j]!=0])/sum(rules[i])
        #                 pr = sum([1 if j in top_attn else 0 for j in range(len(rules[i])) if rules[i][j]!=0])/5
                        
        #                 prs += ['%.6f, %.6f,'%(r, pr)]
        #             print (','.join(prs))
        tags = predictions
        if unsort:
            _, predictions, probs,a,tokens = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs,a,tokens)))]
        return predictions,a,tokens

