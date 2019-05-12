import os
import json
import zipfile
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pprint


def load_data():
    dataset = {
        'train': json.load(zipfile.ZipFile('train.json.zip', 'r').open('train.json')),
        'dev': json.load(zipfile.ZipFile('val.json.zip', 'r').open('val.json')),
        'test': json.load(zipfile.ZipFile('test.json.zip', 'r').open('test.json'))
    }
    return dataset


def read_da(data):
    utts, das_list, span_list = [], [], []
    for id, sess in data.items():
        for turn in sess:
            utts.append(turn['text'])
            das_list.append(turn['dialog_act'])
            span_list.append(turn['span_info'])
    return utts, das_list, span_list


def da_analyse(das_list, span_list):
    da_slot = []
    da_slot_span = []
    for das, span in zip(das_list, span_list):
        for da, svs in das.items():
            for s,v in svs:
                da_slot.append((da,s))
        for da, s, v, start, end in span:
            da_slot_span.append((da,s))
    da_slot = dict(Counter(da_slot))
    da_slot_span = dict(Counter(da_slot_span))
    pprint.pprint(len(da_slot))
    pprint.pprint(len(da_slot_span))
    for k in da_slot:
        if k in da_slot_span:
            print(k, da_slot_span[k]/da_slot[k], da_slot[k], da_slot_span[k])

def get_stats(utts,das):
    intent = []
    intent_slot = []
    sen_num, intent_num, intent_slot_num = [], [], []
    for utt, da in zip(utts, das):
        sen_num.append(len([x for x in utt.split('.') if x])) # delete ''
        intent.extend(list(da.keys()))
        intent_num.append(len(da))
        slot_num = 0
        for k, svs in da.items():
            for s, v in svs:
                slot_num+=1
                intent_slot.append(k+'+'+s)
        intent_slot_num.append(slot_num)
    intent = Counter(intent)
    intent_slot = Counter(intent_slot)
    noise_sen_num = [x+np.random.normal(0, 0.1, 1)[0] for x in sen_num]
    noise_intent_num = [x+np.random.normal(0, 0.1, 1)[0] for x in intent_num]
    noise_intent_slot_num = [x + np.random.normal(0, 0.1, 1)[0] for x in intent_slot_num]
    sen_num2intent_num = np.zeros((10,5))
    for x,y in zip(sen_num, intent_num):
        sen_num2intent_num[x][y] += 1
    s = np.sum(sen_num2intent_num,axis=-1,keepdims=True)
    sen_num2intent_num = sen_num2intent_num/s*100
    print(s)
    print(np.sum(sen_num2intent_num,axis=0))
    print(sen_num2intent_num)
    # plt.bar(Counter(intent_num).keys(),Counter(intent_num).values())
    # plt.scatter(noise_sen_num, noise_intent_slot_num, s=1)
    # plt.bar(list(range(len(intent_slot)))[:50],sorted(intent_slot.values(),reverse=True)[:50])
    # plt.show()


if __name__ == '__main__':
    dataset = load_data()
    utts, das_list, span_list = read_da(dataset['train'])
    da_analyse(das_list, span_list)
