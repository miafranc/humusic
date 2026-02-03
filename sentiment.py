import codecs
import re
import numpy as np
import os

from utils import load_json, save_json, word_tokenizer
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
            senti += idf[w] if pos_words.get(w, -1) != -1 else 0
            senti -= idf[w] if neg_words.get(w, -1) != -1 else 0
    else:
        for w in words:
            senti += 1 if pos_words.get(w, -1) != -1 else 0
            senti -= 1 if neg_words.get(w, -1) != -1 else 0
    return senti


if __name__ == '__main__':
    pos_words, neg_words = pos_neg_words()
    # print(pos_words)
    idf = idf_calculator(False)
    p = get_polarity('nem nem soha', pos_words, neg_words, idf)
    print(p)
