"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder, BERTgenerator
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
        self.generator.load_state_dict(checkpoint['generator'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        device = self.opt['device']
        self.opt = checkpoint['config']
        self.opt['device'] = device

    def save(self, filename):
        params = {
                'generator': self.generator.state_dict(),
                'encoder': self.encoder.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda, device, num_class):
    rules = None
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[i].to('cuda') for i in range(3)]
            labels = Variable(F.one_hot(batch[-1].cuda(), num_classes=num_class)).float()
    else:
        inputs = [Variable(batch[i]) for i in range(3)]
        labels = Variable(F.one_hot(batch[-1], num_classes=num_class)).float()
    return inputs, labels

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.generator = BERTgenerator()
        self.encoder = BERTencoder(opt['num_class'])
        self.criterion = nn.CrossEntropyLoss()

        param_optimizer = list(self.generator.named_parameters())+list(self.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                self.encoder.cuda()
                self.generator.cuda()
                self.criterion.cuda()

        self.optimizer = BertAdam(optimizer_grouped_parameters,
             lr=opt['lr'],
             warmup=opt['warmup_prop'],
             t_total=opt['steps'])

        for param in self.encoder.model.parameters():
            param.requires_grad = False

    def update(self, batch, epoch):
        selection_lambda = 1
        continuity_lambda = 1

        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'], self.opt['num_class'])

        # step forward
        self.encoder.train()
        self.generator.train()

        rationale = self.generator(inputs)
        logits = self.encoder(inputs, rationale)
        loss = self.criterion(logits, labels) + selection_lambda*(torch.sum(rationale)) + continuity_lambda*(torch.sum(rationale[:, 1:]  - rationale[:, :-1]))
        loss_val = loss.item()
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = logits = inputs = labels = None
        return loss_val, self.optimizer.get_lr()[0]

    def predict(self, batch, id2label, tokenizer):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'], self.opt['num_class'])
        # forward
        self.encoder.eval()
        self.generator.eval()
        with torch.no_grad():
            rationale = self.generator(inputs, False)
            tagging_max = np.argmax(rationale.squeeze(2).data.cpu().numpy(), axis=1)
            tagging = torch.round(rationale).squeeze(2)
            probs = self.encoder(inputs, tagging)
            
        loss = self.criterion(probs, labels).item()
        # probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = []
        for i, p in enumerate(predictions):
            # if p != 0:
            t = tagging[i].data.cpu().numpy().tolist()
            if sum(t) == 0:
                t[tagging_max[i]] = 1
            tags += [t]
        
        return predictions, loss, tags