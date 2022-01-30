import json
import argparse
import statistics
import numpy as np
from termcolor import colored
import csv

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='output_lime_132_test_best_model_6.json')
parser.add_argument('--top', type=int, default=3)
parser.add_argument('--origin', type=str, default='dataset/tacred/test.json')
args = parser.parse_args()

data_file = args.origin

origin = json.load(open(data_file))

output = json.load(open(args.data))
tagging_scores = list()
with open('tagging.csv', 'w', newline='') as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["relation", "text"])
    writer.writeheader()
for i, item in enumerate(output):
    gold_label = item['gold_label']
    predicted_label = item['predicted_label']
    words = origin[i]['token']
    ss, se = origin[i]['subj_start'], origin[i]['subj_end']
    os, oe = origin[i]['obj_start'], origin[i]['obj_end']

    if predicted_label != "no_relation":
        tagged = item['gold_tags']
        importance = item['predicted_tags']
        if "lime" in args.data:
            top = [words[j] for j in np.array(item['predicted_tags']).argsort()[-args.top:].tolist()]
            importance = [j for j, w in enumerate(words) if w in top]
        elif "greedy" not in args.data and "tagging" not in args.data:
            importance = np.array(item['predicted_tags']).argsort()[-args.top:].tolist()
        tokens = list()
        if "greedy" not in args.data and "tagging" not in args.data:
            pass
        else:
            for w, word in enumerate(words):
                word = convert_token(word)
                if w>=ss and w<=se:
                    tokens.append('<span style="color:blue;">%s</span>'%word)
                elif w>=os and w<=oe:
                    tokens.append('<span style="color:darkorange;">%s</span>'%word)
                elif w in importance:
                    tokens.append('<span style="color:red;">%s</span>'%word)
                else:
                    tokens.append(word)
        if predicted_label != "no_relation":
            print (" ".join(tokens))
            writer.writerows({'relation': predicted_label, 'text': " ".join(tokens)})
        if len(tagged)>0 and gold_label == predicted_label:
            correct = 0
            pred = 0
            for j, t in enumerate(words):
                if j in importance and j in tagged:
                    correct += 1
            if len(importance) > 0:
                r = correct / len(importance)
            else:
                r = 0
            if len(tagged) > 0:
                p = correct / len(tagged)
            else:
                p = 0
            try:
                f1 = 2.0 * p * r / (p + r)
            except ZeroDivisionError:
                f1 = 0
            tagging_scores.append((r, p, f1))

tr, tp, tf = zip(*tagging_scores)

print("rationale result: {:.2f}\t{:.2f}\t{:.2f}".format(statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))