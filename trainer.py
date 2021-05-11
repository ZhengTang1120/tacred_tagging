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
    rules = None
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[0].to('cuda')]
            labels = Variable(batch[1].cuda())
    else:
        inputs = [Variable(batch[0])]
        labels = Variable(batch[1])
    tokens = batch[0]
    return inputs, labels, tokens

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
        inputs, labels, tokens = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

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
        inputs, labels, tokens = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        tokens = tokens.data.cpu().numpy().tolist()
        orig_idx = batch[2]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        o, b_out = self.encoder(inputs)
        a = o.attentions
        a = a[-1]#.data.cpu().numpy()
        h = o.pooler_output
        logits = self.classifier(h)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = predictions
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions

    def predict_text(self, tokens):
        # forward
        self.encoder.eval()
        self.classifier.eval()
        o, b_out = self.encoder([torch.LongTensor(tokens).cuda()])
        h = o.pooler_output
        logits = self.classifier(h)
        probs = F.softmax(logits, 1).detach().cpu().numpy()
        return probs