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

    def forward(self, inputs):
        words = inputs[0]
        mask = inputs[1]
        outputs = self.model(words, attention_mask=mask, output_attentions=True)
        
        return outputs.pooler_output

class BERTclassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_dim = 1024
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)
        self.opt = opt

    def forward(self, h):
        cls_out = self.dropout(h)#pool(h, out_mask.eq(0), type=pool_type)
        logits = self.classifier(cls_out)
        return logits