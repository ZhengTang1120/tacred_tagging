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
        self.pos_emb = nn.Embedding(6, 5, padding_idx=constant.PAD_ID)

    def forward(self, inputs):
        words, masks, ent_pos, subj_pos, obj_pos, subj_type, obj_type = inputs
        outputs = self.model(words)
        h = outputs.last_hidden_state
        h = torch.cat([h, self.pos_emb(ent_pos)], dim=2)
        out = torch.sigmoid(self.classifier(outputs.pooler_output))

        return h, out

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1029
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, h, masks, subj_pos, obj_pos):
        subj_mask, obj_mask = subj_pos.eq(1000).unsqueeze(2), obj_pos.eq(1000).unsqueeze(2)
        
        pool_type = self.opt['pooling']
        out_mask = masks.unsqueeze(2).eq(0) + subj_mask + obj_mask
        cls_out = pool(h, out_mask.eq(0), type=pool_type)
        logits = self.classifier(cls_out)
        return logits


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
