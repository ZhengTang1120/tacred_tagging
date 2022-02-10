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
    if predicted_label == "per:title":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the title of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:top_members/employees":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the top member/employee of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:employee_of":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is the employee of <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "org:alternate_names":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the alternate name of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:country_of_headquarters":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the country of <span style="color:blue;">{subj_type}({subj})</span>\'s headquarter.'
    if predicted_label == "per:countries_of_residence":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the country of <span style="color:blue;">{subj_type}({subj})</span>\'s residence.'
    if predicted_label == "org:city_of_headquarters":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the city of <span style="color:blue;">{subj_type}({subj})</span>\'s headquarter.'
    if predicted_label == "per:cities_of_residence":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the city of <span style="color:blue;">{subj_type}({subj})</span>\'s residence.'
    if predicted_label == "per:age":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the age of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:stateorprovinces_of_residence":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the state or province of <span style="color:blue;">{subj_type}({subj})</span>\'s residence.'
    if predicted_label == "per:origin":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the origin of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:subsidiaries":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the subsidiary of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:parents":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the parent of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:spouse":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the spouse of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:stateorprovince_of_headquarters":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the state or province of <span style="color:blue;">{subj_type}({subj})</span>\'s headquarter.'
    if predicted_label == "per:children":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the child of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:other_family":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the family member of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:alternate_names":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the alternate name of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:members":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the member of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:siblings":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the sibling of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:schools_attended":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the school <span style="color:blue;">{subj_type}({subj})</span> attended.'
    if predicted_label == "per:parents":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the parent of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:date_of_death":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the date of <span style="color:blue;">{subj_type}({subj})</span>\'s death.'
    if predicted_label == "org:member_of":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is the member of <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "org:founded_by":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is founded by <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "org:website":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the website of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:cause_of_death":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the cause of <span style="color:blue;">{subj_type}({subj})</span>\'s death.'
    if predicted_label == "org:political/religious_affiliation":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is a <span style="color:darkorange;">{obj_type}({obj})</span> affiliation.'
    if predicted_label == "org:founded":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is founded in <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "per:city_of_death":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the city of <span style="color:blue;">{subj_type}({subj})</span>\'s death.'
    if predicted_label == "org:shareholders":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the shareholder of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "org:number_of_employees/members":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the number of employees/members <span style="color:blue;">{subj_type}({subj})</span> has.'
    if predicted_label == "per:date_of_birth":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the date of <span style="color:blue;">{subj_type}({subj})</span>\'s birth.'
    if predicted_label == "per:city_of_birth":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the city of <span style="color:blue;">{subj_type}({subj})</span>\'s birth.'
    if predicted_label == "per:charges":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the charge of <span style="color:blue;">{subj_type}({subj})</span>.'
    if predicted_label == "per:stateorprovince_of_death":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the state or province of <span style="color:blue;">{subj_type}({subj})</span>\'s death.'
    if predicted_label == "per:religion":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the <span style="color:blue;">{subj_type}({subj})</span>\'s religion.'
    if predicted_label == "per:stateorprovince_of_birth":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the state or province of <span style="color:blue;">{subj_type}({subj})</span>\'s birth.'
    if predicted_label == "per:country_of_birth":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the country of <span style="color:blue;">{subj_type}({subj})</span>\'s birth.'
    if predicted_label == "org:dissolved":
        return f'<span style="color:blue;">{subj_type}({subj})</span> is dissolved in <span style="color:darkorange;">{obj_type}({obj})</span>.'
    if predicted_label == "per:country_of_death":
        return f'<span style="color:darkorange;">{obj_type}({obj})</span> is the country of <span style="color:blue;">{subj_type}({subj})</span>\'s death.'


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
        if len(importance) > 0 and len(tagged) == 0:
            text = " ".join(tokens)
            if '<span style="color:red;">' in text:
                relation = template(predicted_label, origin[i]['subj_type'], origin[i]['obj_type'], subj, obj)
                gold = template(gold_label, origin[i]['subj_type'], origin[i]['obj_type'], subj, obj)
                if i in [2256, 11221, 8480, 6270, 6008]:
                    print (gold_label, gold)
                    writer2.writerow({'relation': relation, 'text': text, 'subj_type':origin[i]['subj_type'], 'obj_type':origin[i]['obj_type'], 'subj':" ".join(subj), 'obj':" ".join(obj), "gold": gold})
                elif i in [2256, 11221, 8480, 6270, 6008]:
                    print (gold_label, gold)
                    writer.writerow({'relation': relation, 'text': text, 'subj_type':origin[i]['subj_type'], 'obj_type':origin[i]['obj_type'], 'subj':" ".join(subj), 'obj':" ".join(obj), "gold": gold})
            # else:
            #     print (predicted_label, gold_label)
            #     print ([words[im] for im in importance], tagged, importance)
            #     print (text)
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

print("rationale result: {:.2f}\t{:.2f}\t{:.2f}".format(statistics.mean(tr),statistics.mean(tp),statistics.mean(tf)))