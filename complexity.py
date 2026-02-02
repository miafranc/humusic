import io
import os
import gzip
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import matplotlib
from tqdm import tqdm
import numpy as np
import re

from utils import load_json, save_json, plot_bars, colors_from_values
from settings import *


RE_SENT_SPLIT = r'(\r*\n)+'
RE_SENT_SPLIT_COMPILED = re.compile(RE_SENT_SPLIT, re.U)
RE_VOWEL_SPLIT = r'[aáeéiíoóöőuúüű]'
RE_VOWEL_SPLIT_COMPILED = re.compile(RE_VOWEL_SPLIT, re.U)
RE_SPLIT_COMPILED = re.compile(RE_SPLIT, re.U)


def flesch_kincaid(text):
    num_syllables = len(re.split(RE_VOWEL_SPLIT_COMPILED, text.lower())) - 1
    num_sentences = len(list(filter(None, [s.strip() for s in re.split(RE_SENT_SPLIT_COMPILED, text)])))
    num_words = len(list(filter(None, re.findall(RE_SPLIT_COMPILED, text.strip()))))
    return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59


def gzip_complexity(text):
    mem_zip = io.BytesIO()
    text_bytes = bytes(text, 'utf-8')
    with gzip.GzipFile(fileobj=mem_zip, mode='wb') as zf:
        zf.write(text_bytes)
    return len(mem_zip.getvalue()) / len(text_bytes)


def calculate_complexity(ctype='gzip', rewrite=False):
    if rewrite or not os.path.exists(COMPLEXITY_PATH):
        lyrics = load_json(LYRICS_DATASET_PATH)
        complexity = {}
        for k, d in tqdm(lyrics.items(), desc='Calculating complexity'):
            if ctype == 'gzip':
                complexity[k] = gzip_complexity(d['title'] + ' ' + d['lyrics'])
            else:
                complexity[k] = flesch_kincaid(d['title'] + ' ' + d['lyrics'])
        save_json(complexity, COMPLEXITY_PATH)
    else:
        complexity = load_json(COMPLEXITY_PATH)
    
    return complexity


def complexity_plot_1():
    lyrics = load_json(LYRICS_DATASET_PATH)
    complexity = load_json(COMPLEXITY_PATH)

    comp_values = {g:[] for g in GENRES}
    for k, d in lyrics.items():
        for g in d['tags']:
            if comp_values.get(g, -1) != -1:
                comp_values[g].append(complexity[k])

    for g in GENRES:
        print(f'{g:<30} {np.mean(comp_values[g]):.4f} +/- {np.std(comp_values[g]):.4f}')

    print('#'*30)

    # Least complex lyrics:
    comp_sorted = sorted(complexity.items(), key=lambda x: x[1])
    for c in comp_sorted[:10]:
        print(c[0], c[1], lyrics[c[0]]['artist'], lyrics[c[0]]['title'], lyrics[c[0]]['tags'])
    print('-'*30)
    # Most complex lyrics:
    for c in comp_sorted[-10:]:
        print(c[0], c[1], lyrics[c[0]]['artist'], lyrics[c[0]]['title'], lyrics[c[0]]['tags'])

    plot_bars({g:(np.mean(comp_values[g]), np.std(comp_values[g])) for g in GENRES}) 


def complexity_plot_2(xlabel='', ylabel='', color='b'):
    lyrics = load_json(LYRICS_DATASET_PATH)
    complexity = load_json(COMPLEXITY_PATH)

    complexity = [(k, complexity[k]) for k in lyrics.keys()]
    sorted_complexity = sorted(complexity, key=lambda x: x[1])

    X = [sorted_complexity[i][1] for i in range(len(sorted_complexity)) if lyrics[sorted_complexity[i][0]].get('playcount', -1) != -1]
    Y = [int(lyrics[sorted_complexity[i][0]]['playcount']) for i in range(len(sorted_complexity)) if lyrics[sorted_complexity[i][0]].get('playcount', -1) != -1]

    plt.plot(X, Y, marker='.', markersize=15, markerfacecolor=color, markeredgecolor='k', linestyle='None')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


if __name__ == '__main__':
    # ctype = 'F-K'
    ctype = 'gzip'
    complexity = calculate_complexity(ctype, True)
    complexity_plot_1()
    # complexity_plot_2(f'Complexity ({ctype})', 'Popularity (play count)')
