#!/cs/puls/pyenv/shims/python
# -*- coding: utf-8 -*-

import os
import glob
import zipfile
from bs4 import BeautifulSoup
import numpy as np

np.set_printoptions(threshold=float('nan'))


def get_codes(codefile):
    codes = {}
    i = 0
    with open(codefile, 'r') as cf:
        for line in cf:
            if not line.startswith(';'):
                code = line.strip().split('\t')[0]
                codes[code] = i
                i += 1
    return codes


CODEMAP = get_codes('codes/topic_codes.txt')


def get_labels(doc):
    vec = np.zeros(len(CODEMAP), dtype=int)
    bs = BeautifulSoup(doc, 'lxml')
    topics = bs.find('codes', class_='bip:topics:1.0')
    if topics:
        codes = topics.find_all('code')
        for code in codes:
            vec[CODEMAP[code['code']]] = 1
    return vec


def get_doc_labels(corpus_dir):
    pattern = os.path.join(corpus_dir, '*.zip')
    doc_labels = {}
    for zfile in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(zfile, 'r') as zf:
            for xmlfile in zf.namelist():
                with zf.open(xmlfile, 'r') as xf:
                    doc_labels[xmlfile] = get_labels(xf.read())
    vecs = np.empty((len(doc_labels), len(CODEMAP)), dtype=int)
    for i, (doc, label) in enumerate(sorted(doc_labels.items())):
        vecs[i] = label
    return vecs


result = get_doc_labels('test-corpus')
print('result.shape: ', result.shape)
np.savetxt('test_truth.txt', result, fmt='%i')





