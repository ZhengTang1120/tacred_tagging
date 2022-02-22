import json
import argparse
import statistics
import numpy as np
from termcolor import colored
import csv

from matplotlib.colors import LinearSegmentedColormap, rgb2hex

def template(predicted_label, subj_type, obj_type, subj, obj):
    subj = " ".join(subj)
    obj = " ".join(obj)
    if predicted_label == "no_relation":
        return f'There is no relation between <span style="color:blue;">{subj_type}({subj})</span> and <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "Located_In":
        return f'<span style="color:blue;">{subj_type}({subj})</span> locates in <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "Work_For":
        return f'<span style="color:blue;">{subj_type}({subj})</span> works for <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "Kill":
        return f'<span style="color:blue;">{subj_type}({subj})</span> killed <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "OrgBased_In":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is based in <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "Live_In":
        return f'<span style="color:blue;">{subj_type}({subj})</span> lives in <span style="color:darkorange;">{obj_type}({obj})</span>.'


colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list("Custom", colors)

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
parser.add_argument('--out', type=str, default='lime')
args = parser.parse_args()

data_file = args.origin

origin = json.load(open(data_file))

output = json.load(open(args.data))
tagging_scores = list()
outcsv = open(args.out, 'w', newline='')
writer = csv.DictWriter(outcsv, fieldnames = ["relation", "text", "subj_type", "obj_type", "subj", "obj", "gold"])
writer.writeheader()
outcsv2 = open(args.out.replace(".csv","_negatives.csv") , 'w', newline='')
writer2 = csv.DictWriter(outcsv2, fieldnames = ["relation", "text", "subj_type", "obj_type", "subj", "obj", "gold"])
writer2.writeheader()

for i, item in enumerate(output):
    gold_label = item['gold_label']
    predicted_label = item['predicted_label']
    words = origin[i]['token']
    ss, se = origin[i]['subj_start'], origin[i]['subj_end']
    os, oe = origin[i]['obj_start'], origin[i]['obj_end']
    subj = []
    obj = []

    if predicted_label != "no_relation":
        tagged = item['gold_tags']
        importance = item['predicted_tags']
        # args.top = len(tagged) if len(tagged) != 0 else args.top
        if "lime" in args.data:
            top = [words[j] for j in np.array(item['predicted_tags']).argsort()[-args.top:].tolist()]
            importance = [j for j, w in enumerate(words) if w in top]
        elif "greedy" not in args.data and "tagging" not in args.data:
            importance = np.array(item['predicted_tags']).argsort()[-args.top:].tolist()
        tokens = list()
        # if "greedy" not in args.data and "tagging" not in args.data:
        #     for w, word in enumerate(words):
        #         word = convert_token(word)
        #         if w>=ss and w<=se:
        #             tokens.append('<span style="color:blue;">%s</span>'%word)
        #             # if w in importance:
        #             #     tokens.append('<span style="color:blue; border: 2px solid red;">%s</span>'%word)
        #         elif w>=os and w<=oe:
        #             tokens.append('<span style="color:darkorange;">%s</span>'%word)
        #             # if w in importance:
        #             #     tokens.append('<span style="color:darkorange; border: 2px solid red;">%s</span>'%word)
        #         # elif w in importance:
        #         #     tokens.append('<span style="color:red;">%s</span>'%word)
        #         else:
        #             col = rgb2hex(cm((item['predicted_tags'][w]-lower)/(upper-lower)))
        #             tokens.append(t)
        # else:
        for w, word in enumerate(words):
            word = convert_token(word)
            if w>=ss and w<=se:
                tokens.append('<span style="color:blue;">%s</span>'%word)
                subj.append(word)
                # if w in importance:
                #     tokens.append('<span style="color:blue; border: 2px solid red;">%s</span>'%word)
            elif w>=os and w<=oe:
                tokens.append('<span style="color:darkorange;">%s</span>'%word)
                obj.append(word)
                # if w in importance:
                #     tokens.append('<span style="color:darkorange; border: 2px solid red;">%s</span>'%word)
            elif w in importance:
                tokens.append('<span style="color:red;">%s</span>'%word)
            else:
                tokens.append(word)
        # if len(importance) > 0:
        text = " ".join(tokens)
            # if '<span style="color:red;">' in text:
        relation = template(predicted_label, origin[i]['subj_type'], origin[i]['obj_type'], subj, obj)
        gold = template(gold_label, origin[i]['subj_type'], origin[i]['obj_type'], subj, obj)
        if i in [313, 116, 2323, 3371, 1368, 819, 3816, 3492, 3616, 3109, 2316, 139, 2493, 813, 3212, 2447, 750, 955, 1261, 1124, 1868, 1333, 3328, 2029, 1515, 1167, 3120, 1273, 3059, 300, 2995, 283, 3024, 792, 2169, 1667, 3435, 3396, 2809, 2535, 622, 517, 3008, 1563, 510, 865, 862, 332, 3102, 1825]:
            writer2.writerow({'relation': relation, 'text': text, 'subj_type':origin[i]['subj_type'], 'obj_type':origin[i]['obj_type'], 'subj':" ".join(subj), 'obj':" ".join(obj), "gold": gold})
        elif i in [313, 116, 2323, 3371, 1368, 819, 3816, 3492, 3616, 3109, 2316, 139, 2493, 813, 3212, 2447, 750, 955, 1261, 1124, 1868, 1333, 3328, 2029, 1515, 1167, 3120, 1273, 3059, 300, 2995, 283, 3024, 792, 2169, 1667, 3435, 3396, 2809, 2535, 622, 517, 3008, 1563, 510, 865, 862, 332, 3102, 1825]:
            writer.writerow({'relation': relation, 'text': text, 'subj_type':origin[i]['subj_type'], 'obj_type':origin[i]['obj_type'], 'subj':" ".join(subj), 'obj':" ".join(obj), "gold": gold})

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
outcsv.close()
outcsv2.close()

tr, tp, tf = zip(*tagging_scores)

print("rationale result: {:.4f}\t{:.4f}\t{:.4f}".format(statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))
