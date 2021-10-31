import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertModel

from utils.constant import *

class BERTencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("spanbert-large-cased", cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE))

    def forward(self, input_ids, segment_ids, input_mask):

        _, pooled_output = self.model(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        
        return pooled_output

class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        in_dim = 1024
        self.classifier = nn.Linear(in_dim, num_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h):
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits

class Pipeline(nn.Module):
    def __init__(self, num_class):
        self.encoder = BERTencoder()
        self.classifier = Classifier(num_class)

    def forward(self, input_ids, segment_ids, input_mask, label_ids=None):
        pooled_output = self.encoder(input_ids, segment_ids, input_mask, label_ids)
        logits = self.classifier(pooled_output)

        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
