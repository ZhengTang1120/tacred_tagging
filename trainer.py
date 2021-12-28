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

        h,_ = self.encoder(inputs)
        logits = self.classifier(h)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = logits = inputs = labels = None
        return loss_val, self.optimizer.get_lr()

    def predict(self, batch, id2label, tokenizer):
        inputs, labels = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        # forward
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            h,_ = self.encoder(inputs)
            probs = self.classifier(h)
        loss = self.criterion(probs, labels).item()
        # probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        
        return predictions, loss

    def predict_cand(self, inputs, r):
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            h,_ = self.encoder(inputs)
            probs = self.classifier(h)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()

        best = np.argmax(probs.data.cpu().numpy(), axis=0).tolist()[r]
        return best, predictions[best]

    def update_cand(self, inputs, r):

        # step forward
        self.encoder.train()
        self.classifier.train()

        h, _ = self.encoder(inputs)
        logits = self.classifier(h)

        best = np.argmax(probs.data.cpu().numpy(), axis=0).tolist()[r]

        loss = self.criterion(logits[best].unsqueeze(0), r)
        loss_val = loss.item()
        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        h = logits = inputs = labels = None
        return loss_val

    def predict_with_saliency(self, batch0):
        inputs, labels = unpack_batch(batch0, self.opt['cuda'], self.opt['device'])
        self.encoder.eval()
        self.classifier.eval()

        h, embs = self.encoder(inputs)
        embs.retain_grad()

        logits = self.classifier(h)
        probs = F.softmax(logits, 1)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        score_max = probs[0, predictions]
        
        self.encoder.train()
        self.classifier.train()

        self.classifier.dropout.eval()
        score_max.backward()

        saliency, _ = torch.max(embs.grad.data.abs(),dim=2)
        mask = torch.logical_and(inputs[0].gt(0), inputs[0].lt(20)) + inputs[0].eq(101) + inputs[0].eq(102)
        
        saliency = saliency.masked_fill(mask, 0)
        # top3 = saliency.data.cpu().numpy()[0].argsort()[-3:].tolist()
        return predictions, saliency.data.cpu().numpy()[0][1:-1]

    def predict_proba(self, tokens):
        # forward
        self.encoder.eval()
        self.classifier.eval()

        tokens = torch.LongTensor(tokens).cuda().squeeze(2)
        mask = tokens.eq(0).eq(0).long()
        segment_ids = torch.zeros(tokens.size()).long().cuda()
        batch_size = len(tokens)
        inputs = [tokens, mask, segment_ids]

        h, _ = self.encoder(inputs)
        logits = self.classifier(h)
        probs = F.softmax(logits, 1).detach().cpu().numpy()
        return probs
