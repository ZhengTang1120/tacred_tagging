"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder
from generator import Generator
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


def unpack_batch(batch, cuda, device):
    rules = None
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[i].to('cuda') for i in range(3)]
            labels = Variable(batch[-1].cuda())
    else:
        inputs = [Variable(batch[i]) for i in range(3)]
        labels = Variable(batch[-1].cuda())
    return inputs, labels

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder(opt['num_class'])
        self.generator = Generator(opt, self.encoder.model.config)
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

        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.generator.train()

        rationale = self.generator(inputs[0])
        logits = self.encoder(inputs, rationale)

        loss = self.criterion(logits, labels)
        loss_val = loss.item() # I only care about the classification loss for debugging purpose
        selection_cost, continuity_cost = self.generator.loss(rationale)
        loss += self.opt['selection_lambda'] * selection_cost
        loss += self.opt['continuity_lambda'] * continuity_cost

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
        self.generator.eval()
        with torch.no_grad():
            tagging = self.generator(inputs[0])
            tagging_max = np.argmax(tagging.data.cpu().numpy(), axis=1)
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