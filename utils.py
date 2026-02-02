import os
import numpy as np
import torch
import random
import codecs
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
