"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder, BERTclassifier, Tagger
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
        self.tagger.load_state_dict(checkpoint['tagger'])
        device = self.opt['device']
        self.opt = checkpoint['config']
        self.opt['device'] = device

    def save(self, filename, epoch):
        params = {
                'classifier': self.classifier.state_dict(),
                'encoder': self.encoder.state_dict(),
                'tagger': self.tagger.state_dict(),
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
            inputs = [batch[0].to('cuda')] + [Variable(b.cuda()) for b in batch[1:7]]
            labels = Variable(batch[7].cuda())
            rules  = Variable(batch[9]).cuda()
    else:
        inputs = [Variable(b) for b in batch[:7]]
        labels = Variable(batch[7])
        rules  = Variable(batch[9])
    tokens = batch[0]
    subj_pos = batch[3]
    obj_pos = batch[4]
    tagged = batch[-1]
    return inputs, labels, rules, tokens, subj_pos, obj_pos, tagged

class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.tagger = Tagger()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion2 = nn.BCELoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]+ [p for p in self.tagger.parameters() if p.requires_grad]
        if opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                self.encoder.cuda()
                self.tagger.cuda()
                self.classifier.cuda()
                self.criterion.cuda()
        self.optimizer = AdamW(
            self.parameters,
            lr=opt['lr'],
        )
    
    def update(self, batch, epoch):
        inputs, labels, rules, tokens, subj_pos, obj_pos, tagged = unpack_batch(batch, self.opt['cuda'], self.opt['device'])

        # step forward
        self.encoder.train()
        self.classifier.train()
        self.tagger.train()
        self.optimizer.zero_grad()

        loss = 0
        h, b_out = self.encoder(inputs)
        tagging_output = self.tagger(h)
        loss = self.criterion2(b_out, (~(labels.eq(0))).to(torch.float32).unsqueeze(1))
        if epoch <= 5:
            for i, f in enumerate(tagged):
                if f:
                    loss += self.criterion2(tagging_output[i], rules[i].unsqueeze(1).to(torch.float32))
                    logits = self.classifier(h[i], inputs[1][i].unsqueeze(0), inputs[3][i].unsqueeze(0), inputs[4][i].unsqueeze(0))
                    loss += self.criterion(logits, labels.unsqueeze(1)[i])
        else:
            for i, f in enumerate(tagged):
                if f:
                    loss += self.criterion2(tagging_output[i], rules[i].unsqueeze(1).to(torch.float32))
                    logits = self.classifier(h[i], inputs[1][i].unsqueeze(0), inputs[3][i].unsqueeze(0), inputs[4][i].unsqueeze(0))
                    loss += self.criterion(logits, labels.unsqueeze(1)[i])
                elif labels[i] != 0:
                    tag_cands, n = self.tagger.generate_cand_tags(tagging_output[i], self.opt['device'])
                    if n != -1:
                        logits = self.classifier(h[i], tag_cands, torch.cat(n*[inputs[3][i].unsqueeze(0)], dim=0), torch.cat(n*[inputs[4][i].unsqueeze(0)], dim=0))
                        best = np.argmax(logits.data.cpu().numpy(), axis=0).tolist()[labels[i]]
                        # loss += self.criterion2(tagging_output[i], tag_cands[best].unsqueeze(1).to(torch.float32))
                        loss += self.criterion(logits[best].unsqueeze(0), labels.unsqueeze(1)[i])
        if loss != 0:
            loss_val = loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
        else:
            loss_val = 0
        return loss_val

    def predict(self, batch, id2label, tokenizer, unsort=True):
        inputs, labels, rules, tokens, subj_pos, obj_pos, tagged = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        rules = rules.data.cpu().numpy().tolist()
        tokens = tokens.data.cpu().numpy().tolist()
        orig_idx = batch[8]
        # forward
        self.encoder.eval()
        self.classifier.eval()
        self.tagger.eval()
        h, b_out = self.encoder(inputs)
        tagging_output = self.tagger(h)
        tagging_mask = torch.round(tagging_output).squeeze(2).eq(0)
        tagging = torch.round(tagging_output).squeeze(2)
        logits = self.classifier(h, tagging_mask, inputs[3], inputs[4])
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1) * torch.round(b_out)
        predictions = np.argmax(probs.data.cpu().numpy(), axis=1).tolist()
        tags = []
        subjs = []
        objs = []
        for i, p in enumerate(predictions):
            s = inputs[3].eq(1000).long()[i].data.cpu().numpy().tolist()
            o = inputs[4].eq(1000).long()[i].data.cpu().numpy().tolist()
            subjs += [s]
            objs += [o]
            chunk = inputs[2].eq(4).long()[i].data.cpu().numpy().tolist()
            t = []
            if p != 0:
                t = tagging[i].data.cpu().numpy().tolist()
                if sum(t) == 0:
                    print (tagging_output.size())
                    print (np.argmax(tagging_output.squeeze(2).data.cpu().numpy()))
                tags += [t]
            else:
                t = [0 for x in chunk]
                tags += [[]]
            if sum(rules[i])!=0 and tagged:
                r = sum([1 if t[j]==rules[i][j] else 0 for j in range(len(t)) if rules[i][j]!=0])/sum(rules[i])
                pr = sum([1 if t[j]==rules[i][j] else 0 for j in range(len(t)) if rules[i][j]!=0])/sum(t) if sum(t)!=0 else 0
                r2 = sum([1 if chunk[j]==rules[i][j] else 0 for j in range(len(chunk)) if rules[i][j]!=0])/sum(rules[i])
                pr2 = sum([1 if chunk[j]==rules[i][j] else 0 for j in range(len(chunk)) if rules[i][j]!=0])/sum(chunk) if sum(chunk)!=0 else 0
                print ('%.6f, %.6f, %.6f, %.6f'%(r, pr, r2, pr2))
            
        if unsort:
            _, predictions, probs, tags, rules, tokens, subjs, objs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs, tags, rules, tokens, subjs, objs)))]
        return predictions, tags, rules, tokens, subjs, objs