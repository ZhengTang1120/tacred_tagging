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
from transformers.optimization import get_linear_scheduler_with_warmup

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
            inputs = [batch[i].to('cuda') for i in range(3)]
            labels = Variable(batch[-2].cuda())
    else:
        inputs = [Variable(batch[i]) for i in range(3)]
        labels = Variable(batch[-2])
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

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=opt['lr'],
            weight_decay=0.01
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=opt['steps']*opt['warmup_prop'], 
            num_training_steps=opt['steps']
        )
    
    def update(self, batch, epoch):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        loss = 0
        h = self.encoder(inputs)
        logits = self.classifier(h)
        loss += self.criterion(logits, labels)
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
        else:
            loss_val = 0
        self.scheduler.step()
        h = logits = inputs = labels = None
        return loss_val

    def predict(self, batch, id2label, tokenizer, unsort=True):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        orig_idx = batch[-1]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        h = self.encoder(inputs)
        logits = self.classifier(h)
        loss = self.criterion(logits, labels).item()
        probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, loss