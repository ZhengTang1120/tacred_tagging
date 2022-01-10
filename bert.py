import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel

from utils import constant, torch_utils

class BERTencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("spanbert-large-cased")

    def forward(self, inputs):
        words = inputs[0]
        mask = inputs[1]
        segment_ids = inputs[2]
        h, pooled_output = self.model(words, segment_ids, mask, output_all_encoded_layers=False)
        subj_mask = words.unsqueeze(2).eq(1)
        obj_mask = words.unsqueeze(2).eq(2)
        for i, x in enumerate(torch.sum(subj_mask, 1)):
            if x[0].item() == 0:
                print (words[i])
        for i, x in enumerate(torch.sum(obj_mask, 1)):
            if x[0].item() == 0:
                print (words[i])
        return h, pooled_output, subj_mask, obj_mask

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024
        self.classifier = nn.Linear(3 * in_dim, opt['num_class'])
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)
        self.opt = opt

    def forward(self, h, c, subj_mask, obj_mask):
        cls_out = torch.cat([c, pool(h, subj_mask.eq(0), type="avg"), pool(h, obj_mask.eq(0), type="avg")], 1)
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)
        return logits

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        # print ('size: ', (mask.size(1) - mask.float().sum(1)))
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)