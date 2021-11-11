import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from transformers import BertModel

from utils import constant, torch_utils

class BERTencoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1024
        self.model = BertModel.from_pretrained("SpanBERT/spanbert-large-cased")
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, inputs):
        words = inputs[0]
        mask = inputs[1]
        segment_ids = inputs[2]
        seq_length = words.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=words.device)
        position_ids = position_ids.unsqueeze(0).expand_as(words)
        outputs = self.model(words, attention_mask=mask, token_type_ids=segment_ids, position_ids=position_ids)
        
        h = outputs.last_hidden_state
        out = torch.sigmoid(self.classifier(outputs.pooler_output))

        return h, out

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)
        self.opt = opt

    def forward(self, h, words, tags):
        pool_type = self.opt['pooling']
        h = self.dropout(h)
        out_mask = tags.unsqueeze(2).eq(1) + torch.logical_and(words.unsqueeze(2).gt(0), words.unsqueeze(2).lt(20))
        cls_out = pool(h, out_mask.eq(0), type=pool_type)
        logits = self.classifier(cls_out)
        return logits

class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1024

        self.tagger = nn.Linear(in_dim, 1)
        self.threshold1 = 0.8
        self.threshold2 = 0.2

    def forward(self, h):

        tag_logits = torch.sigmoid(self.tagger(h))
        
        return tag_logits

    def generate_cand_tags(self, tag_logits, device):
        cand_tags = [[]]
        for t in tag_logits:
            if t < self.threshold1 and t > self.threshold2:
                temp = []
                for ct in cand_tags:
                    temp.append(ct+[0])
                    ct.append(1)
                cand_tags += temp
                if len(cand_tags) > 2048:
                    return None, -1
            elif t > self.threshold1:
                for ct in cand_tags:
                    ct.append(1)
            else:
                for ct in cand_tags:
                    ct.append(0)
        with torch.cuda.device(device):
            return torch.BoolTensor(cand_tags).cuda(), len(cand_tags)

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        print (h.size(), mask.size())
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)