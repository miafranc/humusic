from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize, binarize
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection._split import StratifiedKFold, KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import re
import codecs
import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pprint import pprint
import random
from collections import Counter

import argparse
import pickle

from utils import (load_json, load_stopwords, set_seed, conf_matrix)
from settings import *


RE_SPLIT_COMPILED = re.compile(RE_SPLIT, re.U)


def word_tokenizer(text, lowercase=True):
    if lowercase:
        tokens = [str.lower(x) for x in re.findall(RE_SPLIT_COMPILED, text)]
    else:
        tokens = [x for x in re.findall(RE_SPLIT_COMPILED, text)]
    return tokens


def filter_genres(lyrics):
    '''Filter out the genres of RnB, Country and Reggae (not present in GENRES).
    '''
    genres = {g:1 for g in GENRES}
    
    lyrics_new = {}
    for k, v in lyrics.items():
        lyrics_new[k] = lyrics[k]
        tags = v['tags']
        for t in v['tags']:
            if genres.get(t, -1) == -1:
                tags.remove(t)
        if len(tags) == 0:
            del lyrics_new[k]
        else:
            lyrics_new[k]['tags'] = tags

    return lyrics_new


def stratified_multilabel_splits(data, n_splits=5):
    '''Performs a stratified k-fold split by taking into account all the labels
    of a sample, but afterwards leaving a sample (index) with multiple labels only once
    in each fold. However, the same example can appear in multiple folds.

    For example:
    >>> X = {
    >>>     0: {'tags': ['a', 'b', 'c']},
    >>>     1: {'tags': ['a']},
    >>>     2: {'tags': ['b']},
    >>>     3: {'tags': ['a', 'c']}
    >>> }
    >>> Z = stratified_multilabel_splits(X, 3)
    >>> pprint(Z)
    can result in [[[0, 2, 3], [0, 1, 3]], [[0, 1, 3], [2, 3]], [[0, 1, 2, 3], [0]]].
    '''
    labels = []
    indices = []
    for i, d in enumerate(data.values()):
        for g in d['tags']:
            labels.append(g)
            indices.append(i)

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True).split(X=[0]*len(labels), y=labels)
    splits_ok = []
    for s in splits:
        ss = []
        for fold in s:
            ss.append(list(set([indices[i] for i in fold])))
        splits_ok.append(ss)

    return splits_ok


# For aggregating scores for every class:
SCORES_PER_CLASS = []


def scoring_function_macro(x, y):
    return f1_score(x, y, average='macro')


def scoring_function_micro(x, y):
    per_class_scores = f1_score(x, y, average=None)
    # print(per_class_scores)
    SCORES_PER_CLASS.append(per_class_scores)
    return f1_score(x, y, average='micro')


def tc_multiclass():
    data = load_json(LYRICS_DATASET_PATH)
    data = filter_genres(data)

    stopw = load_stopwords(STOPWORDS_PATH)
    
    count_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer, token_pattern=None, 
                                       ngram_range=(NGRAM_MIN, NGRAM_MAX), stop_words=stopw, norm=None)

    train_X = count_vectorizer.fit_transform([d['title'] + ' ' + d['lyrics'] for d in data.values()])
    
    mlb = MultiLabelBinarizer()
    train_y = mlb.fit_transform([d['tags'] for d in data.values()])

    print(f'Classes in order = {mlb.classes_}')

    train_X = SelectPercentile(chi2, percentile=CHI2_PERCENTILE).fit_transform(train_X, train_y)

    clf = MultinomialNB()

    scoring_functions = [scoring_function_micro,
                         scoring_function_macro]

    scores = cross_validate(OneVsRestClassifier(clf), train_X, train_y, 
                            cv=stratified_multilabel_splits(data, n_splits=N_SPLITS), 
                            scoring={sf.__name__:make_scorer(sf) for sf in scoring_functions}, 
                            return_train_score=False,
                            verbose=0,
                            n_jobs=1)
    pprint(scores)
    for s in scores.keys():
        if s != 'fit_time' and s != 'score_time':
            print("\t{:20}: {:.6f} (+/- {:.6f})".format(s[5:], scores[s].mean(), scores[s].std()))
    
    scores2 = np.vstack(SCORES_PER_CLASS)
    print(f'Mean = {np.mean(scores2, axis=0)}')
    print(f'Std = {np.std(scores2, axis=0)}')


def get_2_classes(y, y1, y2):
    '''Returns only those indices that belong to only one class.
    '''
    return [i for i in range(len(y)) if (y1 in y[i]) ^ (y2 in y[i])]


def tc_one_vs_one():
    gen = GENRES
    gen2ind = {gen[i]:i for i in range(len(gen))}

    M = np.zeros((len(gen), len(gen)))

    data = load_json(LYRICS_DATASET_PATH)
    stopw = load_stopwords(STOPWORDS_PATH)

    for i in range(len(gen)-1):
        for j in range(i+1, len(gen)):
            y1, y2 = gen[i], gen[j]
            
            count_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer, token_pattern=None,
                                               binary=False, ngram_range=(NGRAM_MIN, NGRAM_MAX), stop_words=stopw)

            train_X = count_vectorizer.fit_transform([d['title'] + ' ' + d['lyrics'] for d in data.values()])
            train_y = [d['tags'] for d in data.values()]
            
            c2 = get_2_classes([d['tags'] for d in data.values()], y1, y2)
            print(Counter([y1 if y1 in train_y[idx] else y2 for idx in c2]))

            train_X = train_X[c2]
            train_y_2 = [1 if y1 in train_y[i] else -1 for i in c2]
            train_y = train_y_2

            train_X = SelectPercentile(chi2, percentile=CHI2_PERCENTILE).fit_transform(train_X, train_y)

            clf = MultinomialNB()

            scoring_functions = [f1_score]

            cv = KFold(n_splits=N_SPLITS, shuffle=True)
            scores = cross_validate(clf, train_X, train_y, 
                                    cv=cv, 
                                    scoring={sf.__name__:make_scorer(sf) for sf in scoring_functions}, 
                                    return_train_score=False,
                                    verbose=0,
                                    n_jobs=1)
            for s in scores.keys():
                if s != 'fit_time' and s != 'score_time':
                    print("\t{:20}: {:.6f} (+/- {:.6f})".format(s[5:], scores[s].mean(), scores[s].std()))
                    I = gen2ind[gen[i]]
                    J = gen2ind[gen[j]]
                    if I > J:
                        aux = I
                        I = J
                        J = aux
                    M[I][J] = scores[s].mean()

            print('#'*30)

    # Hardcoded; used just for safety:
    with codecs.open('data/ovo.pickle', 'wb') as f:
        pickle.dump(M, f)
    
    conf_matrix(M, GENRES)


if __name__ == '__main__':
    set_seed(SEED)

    # with codecs.open('data/ovo.pickle', 'rb') as f:
    #     M = pickle.load(f)
    # conf_matrix(M, GENRES)
    # exit(0)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--multiclass', required=False, action='store_true')
    group.add_argument('--ovo', required=False, action='store_true')

    args = parser.parse_args()
    
    if args.multiclass:
        tc_multiclass()
    elif args.ovo:
        tc_one_vs_one()
