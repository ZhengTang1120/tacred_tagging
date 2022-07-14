import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel

from utils import constant, torch_utils

class BERTencoder(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        in_dim = 1024 
        self.model = BertModel.from_pretrained("spanbert-large-cased")
        self.classifier = nn.Linear(in_dim, num_class)
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)

    def forward(self, inputs, rationale_mask):
        words = inputs[0]
        mask = inputs[1]
        segment_ids = inputs[2]

        subj_mask = torch.logical_and(words.unsqueeze(2).gt(4), words.unsqueeze(2).lt(9))
        obj_mask = torch.logical_and(words.unsqueeze(2).gt(0), words.unsqueeze(2).lt(5))
        entity_mask = subj_mask + obj_mask
        rationale_mask.masked_fill(entity_mask.squeeze(2), 1) # Force to set subject and object as important features

        h, pooled_output = self.model(words, segment_ids, rationale_mask, output_all_encoded_layers=False)
        # cls_out = torch.cat([pool(h, subj_mask.eq(0), type="avg"), pool(h, obj_mask.eq(0), type="avg")], 1)
        cls_out = self.dropout(pooled_output)
        logits = self.classifier(cls_out)
        return logits

def pool(h, mask, type):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        # print ('size: ', (mask.size(1) - mask.float().sum(1)))
        return torch.nan_to_num(h.sum(1) / (mask.size(1) - mask.float().sum(1)))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
