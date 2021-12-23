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
        h, pooled_output, embeddings = self.model(words, segment_ids, mask, output_all_encoded_layers=False)
        return pooled_output, embeddings

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