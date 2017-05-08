# coding=utf-8
#!/usr/bin/env python

import os
import re
from sklearn.utils import shuffle
import numpy as np

from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

def read_reviews(path, data_type="train", labels=['pos', 'neg'], shuffle_read=True, partial_read=False):
    
    # Parameter validation.
    if not data_type in ['train', 'test']:
        print("data_type %s is not valid" %str(data_type))
        return None
        
    for l in labels:
        if not l in ['pos', 'neg', 'unsup']:
            print ("label %s is not valid" %str(l))
            return None
    
    data_path = [path + "/" + data_type + "/" + label + "/" for label in labels]
    
    labels = []
    texts = []

    for p in data_path:
        t, l = read_all_from_path(p)
        texts += t
        labels += l
    
    if shuffle_read:
        sh_texts, sh_labels = shuffle(texts, labels)
        return sh_texts, sh_labels
    return texts, labels
    
    
def read_all_from_path(path, partial_read=False):
    # recursively read from text files from give directory.
    texts = []
    labels = []
    for f in os.listdir(path):
        try:
            name, ext = f.split(".")
            star = int(name.split("_")[-1])
            if ext != "txt":
                continue
            with open(path + f, "r") as rf:
                texts.append(rf.read().lower())
                if star != 0:
                    labels.append('neg' if star < 5 else 'pos')
                else:
                    labels.append('unsup')
        except Exception as e:
            print(e)
            continue
    
    return texts, labels


def read_vocabulary(path):
    # read vocabulary list from provided text file.
    with open(path, 'r') as f:
        vocabulary = f.read()
    return vocabulary.split('\n')


def tokenize(s):
    # Tokenizer string with html tags and prunctuations removed.
    s = re.sub(r'</?\w+\s?/?>', '', s)
    tokenizer = RegexpTokenizer(r'\w{2,}')
    tokens = tokenizer.tokenize(s)
    return tokens


def read_to_word_idx(path, data_type="train", labels=['pos', 'neg'], shuffle_read=True, partial_read=False, n_words=500):
    
    data, label = read_reviews(path, data_type, labels, shuffle_read, partial_read)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    X = tokenizer.texts_to_sequences(data)
    
    oov_char = 2
    X = [[oov_char if w >= n_words else w for w in x] for x in X]
    
    X = np.array(X)

    le = LabelBinarizer()
    le.fit(labels)
    y = le.transform(label)
    y = np.array([l[0] for l in y])
    
    return X, y
