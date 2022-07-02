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
    if cuda:
        with torch.cuda.device(device):
            inputs = [batch[i].to('cuda') for i in range(4)]
            labels = Variable(batch[-1].cuda())
    else:
        inputs = [Variable(batch[i]) for i in range(4)]
        labels = Variable(batch[-1])
    return inputs, labels, batch[4]

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

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
             t_total= opt['train_batch'] * self.opt['num_epoch'])

    def update(self, batch, epoch):
        inputs, labels, has_tag = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        selection_lambda = 1
        continuity_lambda = 1
        # step forward
        self.encoder.train()
        self.classifier.train()

        h = self.encoder(inputs)
        logits, rationale = self.classifier(h, inputs[0])
        loss = self.criterion(logits, labels) + selection_lambda*(torch.sum(rationale)) + continuity_lambda*(torch.sum(rationale[:, 1:]  - rationale[:, :-1]))
        loss_val = loss.item()

        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = b_out = logits = inputs = labels = None
        return loss_val, self.optimizer.get_lr()[0]

    def predict(self, batch, id2label, tokenizer):
        inputs, labels, has_tag = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        # forward
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            h = self.encoder(inputs)
            probs, rationale = self.classifier(h, inputs[0])
            tagging_max = np.argmax(rationale.squeeze(2).data.cpu().numpy(), axis=1)
            tagging = torch.round(rationale).squeeze(2)
        loss = self.criterion(logits, labels).item()
        for i, f in enumerate(has_tag):
            if f:
                loss += self.criterion2(tagging_output[i], inputs[3][i].unsqueeze(1).to(torch.float32))
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = []
        for i, p in enumerate(predictions):
            if p != 0:
                t = tagging[i].data.cpu().numpy().tolist()
                if sum(t) == 0:
                    t[tagging_max[i]] = 1
                tags += [t]
            else:
                tags += [[]]
        return predictions, tags, loss
        

