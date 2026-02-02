import os
import numpy as np
import torch
import random
import codecs
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(fname):
    with codecs.open(fname, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fname):
    with codecs.open(fname, 'w') as f:
        json.dump(data, f, indent=2)


def conf_matrix(M, labels, xlabel='', ylabel=''):
    disp = ConfusionMatrixDisplay(confusion_matrix=M, 
                                  display_labels=[labels[i] for i in range(len(labels))])
    disp.plot(values_format=".2f", cmap="YlOrRd", colorbar=False)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.show()


def load_stopwords(fname):
    f = codecs.open(fname, 'r')
    stopw = f.readlines()
    f.close()
    return [s.strip() for s in stopw]


def colors_from_values(values, palette_name):
    # cmap = cm.get_cmap(palette_name)
    cmap = matplotlib.colormaps[palette_name]
    normalized = (values - min(values)) / (max(values) - min(values))
    return np.array([cmap(v) for v in normalized])


def plot_bars(res):
    sorted_scores = sorted(res.items(), key=lambda x: x[1][0])
    y_pos = range(len(sorted_scores))
    
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    ax.grid(visible=True, which='both', axis='x', color='white', linewidth=1)
    ax.barh(y_pos, [s[1][0] for s in sorted_scores], xerr=[s[1][1] for s in sorted_scores], 
            capsize=4, edgecolor='grey', 
            color=colors_from_values(np.array([s[1][0] for s in sorted_scores]), 'YlOrRd'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in sorted_scores])
    
    ax.set_facecolor('lavender')
    plt.show()
