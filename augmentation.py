import nlpaug.augmenter.word as naw
import json
import random
import secrets
from math import floor
from copy import deepcopy

with open('/Users/mehmetyildiz/hub/toqad/data/toqad-train.json',encoding='utf-8') as f:
    data = json.load(f)


back_translation_aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-tc-big-tr-en',
    to_model_name='Helsinki-NLP/opus-mt-tc-big-en-tr'
    )

x = 0
for i in range(0,len(data["data"])):
    for j in range(0,len(data["data"][i]["paragraphs"])):
        for k in range(0,len(data["data"][i]["paragraphs"][j]["qas"])):
            tmp_in = data["data"][i]["paragraphs"][j]["qas"][k]["question"]
            print("-"*20 + str(x))
            print('Og is',tmp_in)
            tmp_out_list = back_translation_aug.augment(tmp_in)
            print('before',len(data["data"][i]["paragraphs"][j]["qas"]))
            print(tmp_out_list)
            if tmp_out_list != '':
                dict = deepcopy(data["data"][i]["paragraphs"][j]["qas"][k])
                dict["id"] = int(secrets.randbelow(1500000))
                dict["question"] = tmp_out_list #.replace("'", '"')
                data["data"][i]["paragraphs"][j]["qas"].append(dict)
                print('after',len(data["data"][i]["paragraphs"][j]["qas"]))
                print('paragraph number is', j)
            x = x + 1

print('Writing Final JSON')
with open('train-v2q.json', 'w') as outfile:
    json.dump(data, outfile)  


