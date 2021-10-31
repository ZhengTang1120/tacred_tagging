from utils.scorer import *
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random
import logging

from bert import *
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data import *

from torch.utils.data import DataLoader, TensorDataset

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(model, device, eval_dataloader, eval_label_ids, id2label, verbose=True, logger=None):
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
    
    prec, recall, f1 = score(eval_labels, [id2label[i] for i in preds], verbose)
    result = {'precision': prec, 'recall': recall, 'f1': f1}
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return preds, result

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args()

    processor = DataProcessor()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_list = processor.get_labels(args.data_dir, logger)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)

    special_tokens = {w : "[unused%d]" % (i + 1) for i, w in enumerate(ENTITY_TOKENS)}

    if args.eval_test:
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, logger)
    logger.info("***** Test *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    eval_label_ids = all_label_ids

    model = Pipeline(num_labels)
    model.to(device)
    preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, id2label, True, logger)
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for ex, pred in zip(eval_examples, preds):
            f.write("%s\t%s\n" % (ex.guid, id2label[pred]))
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key in sorted(result.keys()):
            f.write("%s = %s\n" % (key, str(result[key])))
