import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel

from utils import constant, torch_utils

class BERTgenerator(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1024 
        self.model = BertModel.from_pretrained("spanbert-large-cased")
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)
        self.generator = nn.Linear(in_dim, 1)

    def forward(self, inputs, is_train):
        words = inputs[0]
        mask = inputs[1]
        segment_ids = inputs[2]
        h, pooled_output = self.model(words, segment_ids, mask, output_all_encoded_layers=False)

        rationale = torch.sigmoid(self.generator(F.relu(self.dropout(h))))
        return rationale if is_train else torch.round(rationale)

class BERTencoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024 
        self.model = BertModel.from_pretrained("spanbert-large-cased")
        self.classifier = nn.Linear(in_dim * 3, opt['num_class'])
        
        self.opt = opt

    def forward(self, inputs, rationale, subj_mask, obj_mask):
        words = inputs[0]
        mask = inputs[1]
        segment_ids = inputs[2]
        h, pooled_output = self.model(words, segment_ids, mask, output_all_encoded_layers=False)
        
        cls_out = torch.cat([pool(h, rationale_mask.eq(0), type="avg"), pool(h, subj_mask.eq(0), type="avg"), pool(h, obj_mask.eq(0), type="avg")], 1)
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