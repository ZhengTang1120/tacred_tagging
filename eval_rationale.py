import json
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='output_lime_667_test_best_model_8.json')
args = parser.parse_args()

data_file = '../../tacred_tagging/dataset/tacred/conll04_test_tacred.json'

origin = json.load(open(data_file))

output = json.load(open(args.data))
tagging_scores = list()
for i, item in enumerate(output):
    gold_label = item['gold_label']
    predicted_label = item['predicted_label']
    words = origin[i]['token']

    if predicted_label != "no_relation":
        tagged = item['gold_tags']
        importance = item['predicted_tags']
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