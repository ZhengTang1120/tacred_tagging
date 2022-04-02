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

from pytorch_pretrained_bert.optimization import BertAdam

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

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

    def save(self, filename):
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
            inputs = [batch[i].to('cuda') for i in range(3)]
            labels = Variable(batch[-1].cuda())
    else:
        inputs = [Variable(batch[i]) for i in range(3)]
        labels = Variable(batch[-1])
    return inputs, labels

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.criterion = nn.CrossEntropyLoss()

        param_optimizer = list(self.classifier.named_parameters())+list(self.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]
        if opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                self.encoder.cuda()
                self.classifier.cuda()
                self.criterion.cuda()

        self.optimizer = BertAdam(optimizer_grouped_parameters,
             lr=opt['lr'],
             warmup=opt['warmup_prop'],
             t_total=opt['steps'])

    def update(self, batch, epoch):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()

        h = self.encoder(inputs)
        subj_mask = torch.logical_and(inputs[0].gt(0), inputs[0].lt(3))
        obj_mask = torch.logical_and(inputs[0].gt(2), inputs[0].lt(20))
        logits = self.classifier(h, subj_mask, obj_mask)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = logits = inputs = labels = None
        return loss_val, self.optimizer.get_lr()[0]

    def predict(self, batch, id2label, tokenizer):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        # forward
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            h = self.encoder(inputs)
            subj_mask = torch.logical_and(inputs[0].unsqueeze(2).gt(0), inputs[0].unsqueeze(2).lt(3))
            obj_mask = torch.logical_and(inputs[0].unsqueeze(2).gt(2), inputs[0].unsqueeze(2).lt(20))
            probs = self.classifier(h, subj_mask, obj_mask)
        loss = self.criterion(probs, labels).item()
        # probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        
        return predictions, loss