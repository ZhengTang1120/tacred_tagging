import argparse
import random
import time
import json
import logging

from data import *
from utils.constant import *

from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main(args):
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
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    args = parser.parse_args()
    main(args)