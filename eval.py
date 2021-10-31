from utils.scorer import *
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def evaluate(model, device, eval_dataloader, eval_label_ids, id2label, verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, len(id2label)), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    eval_labels = [id2label[i] for i in eval_label_ids.numpy()]
    preds = np.argmax(preds[0], axis=1)
    
    prec, recall, f1 = score(eval_labels, [id2label[i] for i in preds], verbose=True)

    return preds, {'precision': prec, 'recall': recall, 'f1': f1}