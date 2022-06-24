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
        return h

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024 
        self.classifier = nn.Linear(in_dim * 3, opt['num_class'])
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)
        self.generator = nn.Linear(in_dim * 2, 1)
        self.opt = opt

    def forward(self, h, subj_mask, obj_mask):
        h2 = torch.cat([pool(h, subj_mask.eq(0), type="avg"), pool(h, obj_mask.eq(0), type="avg")], 1)
        rationale = torch.sigmoid(self.generator(F.relu(self.dropout(h2))))
        rationale_mask = torch.round(rationale)
        print (h2.size(), h.size(), rationale_mask.size())
        cls_out = torch.cat([pool(h, rationale_mask.eq(0), type="avg"), h2], 1)
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)
        return logits, rationale_mask

def pool(h, mask, type):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        # print ('size: ', (mask.size(1) - mask.float().sum(1)))
        return torch.nan_to_num(h.sum(1) / (mask.size(1) - mask.float().sum(1))) # # solution for masked all tokens, not sure if there is a better solution or not...
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)