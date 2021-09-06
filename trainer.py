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
            inputs = [batch[0].to('cuda')] + [Variable(b.cuda()) for b in batch[1:7]]
            labels = Variable(batch[7].cuda())
    else:
        inputs = [Variable(b) for b in batch[:7]]
        labels = Variable(batch[7])
    tokens = batch[0]
    subj_pos = batch[3]
    obj_pos = batch[4]
    ent_pos = batch[2]
    return inputs, labels, tokens, subj_pos, obj_pos, ent_pos

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion2 = nn.BCELoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]#+ [p for p in self.tagger.parameters() if p.requires_grad]
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
        inputs, labels, tokens, subj_pos, obj_pos, ent_pos = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        loss = 0
        h, b_out = self.encoder(inputs)
        loss = self.criterion2(b_out, (~(labels.eq(0))).to(torch.float32).unsqueeze(1))
        if self.opt['mask'] == 1:
            mask = ent_pos.ge(1)
        else:
            mask = ent_pos.eq(4)
        logits = self.classifier(h, mask, inputs[3], inputs[4])
        loss += self.criterion(logits, labels.long())
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
        else:
            loss_val = 0
        return loss_val

    def predict(self, batch, id2label, tokenizer, unsort=True):
        inputs, labels, tokens, subj_pos, obj_pos, ent_pos = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        tokens = tokens.data.cpu().numpy().tolist()
        orig_idx = batch[8]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        if self.opt['mask'] == 1:
            mask = ent_pos.ge(1)
        else:
            mask = ent_pos.eq(4)
        h, b_out = self.encoder(inputs)
        logits = self.classifier(h, mask, inputs[3], inputs[4])
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1) * torch.round(b_out)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
            
        if unsort:
            _, predictions, probs, tokens, subjs, objs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs, tokens, subjs, objs)))]
        return predictions, tokens, subjs, objs




