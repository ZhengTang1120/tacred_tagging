import argparse
import random
import time
import json
import logging

from data import *
from utils.constant import *
from eval import evaluate

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(args)
    logger.info("device: {}".format(device))

    processor = DataProcessor()

    label_list = processor.get_labels(args.data_dir, logger)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    special_tokens = {w : "[unused%d]" % (i + 1) for i, w in enumerate(ENTITY_TOKENS)}

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, logger)
    logger.info("***** Dev *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size)
    eval_label_ids = all_label_ids

    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, logger)
    train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    train_batches = [batch for batch in train_dataloader]

    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

    logger.info("***** Training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    best_result = None
    eval_step = max(1, len(train_batches) // args.eval_per_epoch)
    lr = args.learning_rate

    model = BertForSequenceClassification.from_pretrained(
            args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_labels=num_labels)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    start_time = time.time()
    global_step = 0
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
        random.shuffle(train_batches)
        for step, batch in enumerate(train_batches):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if (step + 1) % eval_step == 0:
                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                             epoch, step + 1, len(train_batches),
                             time.time() - start_time, tr_loss / nb_tr_steps))
                save_model = False
                preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, id2label)
                model.train()
                result['global_step'] = global_step
                result['epoch'] = epoch
                result['learning_rate'] = lr
                result['batch_size'] = args.train_batch_size
                logger.info("First 20 predictions:")
                for pred, label in zip(preds[:20], eval_label_ids.numpy()[:20]):
                    sign = u'\u2713' if pred == label else u'\u2718'
                    logger.info("pred = %s, label = %s %s" % (id2label[pred], id2label[label], sign))
                if (best_result is None) or (result["f1"] > best_result["f1"]):
                    best_result = result
                    save_model = True
                    logger.info("!!! Best dev f1 (lr=%s, epoch=%d): %.2f" %
                                (str(lr), epoch, result["f1"] * 100.0))
                if save_model:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)
                    if best_result:
                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                            for key in sorted(result.keys()):
                                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--output_dir", default="saved_models_spanbert", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()
    main(args)