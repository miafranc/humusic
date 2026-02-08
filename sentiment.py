import codecs
import re
import numpy as np
import os
import pandas as pd
from collections import Counter
import argparse
from sklearn.metrics import f1_score

from utils import load_json, save_json, word_tokenizer, plot_bars
from settings import *


RE_SPLIT_COMPILED = re.compile(RE_SPLIT, re.U)


def pos_neg_words():
    with codecs.open(SENTI_KAGGLE_POS_PATH, 'r') as f:
        pos_words = f.readlines()
    with codecs.open(SENTI_KAGGLE_NEG_PATH, 'r') as f:
        neg_words = f.readlines()
    pos_words = [w.strip().lower() for w in pos_words]
    neg_words = [w.strip().lower() for w in neg_words]

    with codecs.open(SENTI_PRECO_POS_PATH, 'r') as f:
        data = f.readlines()
    pos_words = set(pos_words).union([w.strip().lower() for w in data])
    with codecs.open(SENTI_PRECO_NEG_PATH, 'r') as f:
        data = f.readlines()
    neg_words = set(neg_words).union([w.strip().lower() for w in data])

    common_words = neg_words.intersection(pos_words)
    neg_words = neg_words.difference(common_words)
    pos_words = pos_words.difference(common_words)

    return {pw:1 for pw in pos_words}, {nw:1 for nw in neg_words}


def idf_calculator(rewrite=False):
    if os.path.exists(IDF_PATH) and not rewrite:
        idf = load_json(IDF_PATH)
        return idf

    lyrics = load_json(LYRICS_DATASET_PATH)
    idf = {}
    for k in lyrics.keys():
        words = word_tokenizer(lyrics[k]['title'] + ' ' + lyrics[k]['lyrics'], RE_SPLIT_COMPILED)
        for w in set(words):
            idf[w] = idf.get(w, 0) + 1
    for w in idf.keys():
        idf[w] = np.log2(len(lyrics) / idf[w])
    
    save_json(idf, IDF_PATH)
    return idf


def get_polarity(text, pos_words, neg_words, idf=None):
    words = word_tokenizer(text, RE_SPLIT_COMPILED)
    senti = 0
    if idf != None:
        for w in words:
            idf_word = idf[w] if idf.get(w, -1) != -1 else 1
            # idf_word = idf[w] if idf.get(w, -1) != -1 else 0
            senti += idf_word if pos_words.get(w, -1) != -1 else 0
            senti -= idf_word if neg_words.get(w, -1) != -1 else 0
    else:
        for w in words:
            senti += 1 if pos_words.get(w, -1) != -1 else 0
            senti -= 1 if neg_words.get(w, -1) != -1 else 0
    return senti


def genre_sentiments(use_idf=True):
    lyrics = load_json(LYRICS_DATASET_PATH)
    pos_words, neg_words = pos_neg_words()
    idf = idf_calculator(False)
    senti = {g:{'pos':0, 'neg':0, 'neu': 0} for g in GENRES}

    for d in lyrics.values():
        p = get_polarity(d['title'] + ' ' + d['lyrics'], pos_words, neg_words, 
                         idf if use_idf else None)
        for g in d['tags']:
            if g in GENRES:
                if p > 0:
                    senti[g]['pos'] += 1
                elif p < 0:
                    senti[g]['neg'] += 1
                else:
                    senti[g]['neu'] += 1

    return senti


def load_opinhubank():
    df = pd.read_csv('data/OpinHuBank_20130106.csv', encoding='utf-8', on_bad_lines='warn')
    data = []
    for index, row in df.iterrows():
        sentence = row['Sentence']
        s_true = (row['Annot1'] + row['Annot2'] + row['Annot3'] + row['Annot4'] + row['Annot5']) / 5.
        label = 0
        if s_true < 0:
            label = -1
        elif s_true > 0:
            label = 1
        data.append([sentence, label])
    print(Counter([d[1] for d in data]))
    return data


def load_husst():
    df = load_json('data/sst_train.json')
    counts = {'pos': 0, 
              'neg': 0, 
              'neu': 0}
    data = [[d['Sent'], 
             1 if d['label'] == 'positive' else -1 if d['label'] == 'negative' else 0] for d in df]
    print(Counter([d[1] for d in data]))
    return data


def validate(threshold=0, dataset='opinhubank'):
    pos_words, neg_words = pos_neg_words()
    # idf = idf_calculator(False)
    idf = None

    data = []
    match dataset:
        case 'opinhubank':
            data = load_opinhubank()
        case 'husst':
            data = load_husst()

    data = [[d[0], 1 if d[1] >= 0 else -1] for d in data]

    y_true = [d[1] for d in data]
    y_pred = []

    acc = 0
    for d in data:
        p = 1 if get_polarity(d[0], pos_words, neg_words, idf) >= threshold else -1
        acc += (p == d[1])
        y_pred.append(p)
    acc /= len(data)

    print(f'Accuracy: {acc:.4f}')
    print(f'F1: {f1_score(y_true, y_pred, average="macro"):.4f}')


if __name__ == '__main__':
    # validate(0, 'opinhubank')
    # validate(0, 'husst')

    parser = argparse.ArgumentParser()
    parser.add_argument('--idf', required=False, action='store_true')

    args = parser.parse_args()

    senti = genre_sentiments(use_idf=args.idf)
    plot_bars({g: (senti[g]['pos'] / senti[g]['neg'], 0) for g in GENRES})
