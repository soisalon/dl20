

import os
import zipfile
import glob
import string
import random

import xmltodict
import nltk
import torch
import numpy as np

from vars import DATA_DIR, PROJ_DIR, TEST_DATA_DIR


class DocDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, params, train=True):

        self.train = train
        self.all = params.final

        data_path = os.path.join(PROJ_DIR, 'dl20', root_dir, 'all.pt')
        labels_path = os.path.join(PROJ_DIR, 'dl20', 'ground_truth.txt')
        test_path = os.path.join(PROJ_DIR, 'dl20', root_dir, 'test.pt')

        data = torch.load(test_path) if not self.train and self.all else torch.load(data_path)
        labels = torch.tensor(np.loadtxt(labels_path))
        n_docs = data.shape[0]
        n_dev = int(n_docs * params.dev_ratio)
        n_tr = n_docs - n_dev

        if self.train and not self.all:
            self.tr_data = data[:n_tr, ...]
            self.tr_labels = labels[:n_tr, ...]
        elif not self.train and not self.all:
            self.dev_data = data[n_tr:, ...]
            self.dev_labels = labels[n_tr:, ...]
        elif self.train and self.all:
            self.tr_data = data
            self.tr_labels = labels
        else:
            self.dev_data = data

    def __len__(self):
        if self.train:
            return len(self.tr_data)
        else:
            return len(self.dev_data)

    def __getitem__(self, idx):
        if self.train:
            item, target = self.tr_data[idx], self.tr_labels[idx]
            return item, target
        elif not self.train and not self.all:
            item, target = self.dev_data[idx], self.dev_labels[idx]
            return item, target


def get_model_savepath(params, ext='.pt'):

    mod = params.model_name[:3]
    encoder = params.emb_pars[0][:4]
    nl = params.n_conv_layers
    ks = '+'.join(params.kernel_shapes)
    pls = '+'.join(params.pool_sizes)
    insh = params.input_shape
    nk = '+'.join([str(n) for n in params.n_kernels])
    caf, faf, oaf = params.conv_act_fn[:3], params.fc_act_fn[:3], params.out_act_fn[:3]
    d = params.dropout

    bs, ne, op, ls = params.batch_size, params.n_epochs, params.optim[:3], params.loss_fn[:3]
    op_pars = '+'.join(params.opt_params) if params.opt_params else 'def'

    hu = '+'.join([str(n) for n in params.h_units])

    return '-'.join(map(str, [mod, encoder, nl, ks, pls, insh, nk, caf, faf, oaf, d, hu,
                                         bs, ne, op, op_pars, ls])) + ext


def get_docs(cum_docs, zips):

    inds = max(list(cum_docs.values()))
    batch_docs = []
    for i in range(inds):
        for zi, zip in enumerate(zips):
            if i < cum_docs[zip]:
                file_i = i if zi == 0 else i - cum_docs[zips[zi - 1]]
                with zipfile.ZipFile(zip, 'r') as zf:
                    with zf.open(zf.namelist()[file_i], 'r') as xf:
                        batch_docs += [xf.read()]
                break

    return batch_docs


def get_cum_docs_per_zip(zips):

    docs_per_zip = {}
    count = 0
    for zipf in zips:
        with zipfile.ZipFile(zipf) as zf:
            count += len(zf.namelist())
            docs_per_zip[zipf] = count
    return docs_per_zip


def get_doc_words(xmlfile):
    """
    Extract words from doc: <title>, <headline>, and <text>, and put the woords into a list.

    Filtering out the following:
    - words shorter than 4 characters
    - non-alphabetic words

    :param xmlfile:
    :param filter:
    :return:
    """

    doc_dict = xmltodict.parse(xmlfile)

    keys1 = list(doc_dict.keys())
    k1 = keys1[0]
    keys2 = [k2 for k1 in doc_dict for k2 in doc_dict[k1]]

    title = doc_dict[k1]['title'] if 'title' in keys2 else ''
    headline = doc_dict[k1]['headline'] if 'headline' in keys2 else ''
    text = doc_dict[k1]['text'] if 'text' in keys2 else None

    sents = text['p'] if text else ''           # text['p'] is a list

    title_tokens = nltk.word_tokenize(title) if title else []
    headline_tokens = nltk.word_tokenize(headline) if headline else []
    sents_tokens = [nltk.word_tokenize(s) for s in sents if s] if sents else []
    sents_tokens = [w for s in sents_tokens for w in s] if sents_tokens else []

    words = title_tokens + headline_tokens + sents_tokens

    words = [w for w in words if w.isalpha()]        # filter out numeric words, they don't predict topic

    words = [w.lower() for w in words]
    words = [w for w in words if not len(w) < 4]        # filter words with length < 4

    if not words:
        print('No words in xmlfile:  ', xmlfile)
        print('doc_dict: ', doc_dict)

    return words


def sample_sequences(word_batch, max_width):

    seqs = []
    for wordlist in word_batch:
        diff = len(wordlist) - max_width
        if diff > 0:
            start = random.randint(0, diff)
            end = start + max_width
            seqs += [wordlist[start:end]]
        else:
            seqs += [wordlist]
    return seqs


if __name__ == '__main__':

    # get words representing newsitems into a text file

    n_classes = 126
    # n_docs = 299773  # docs (xml files) in total
    pattern = os.path.join(TEST_DATA_DIR, '*.zip')
    zips = sorted(glob.glob(pattern))  # for reading input batches
    cum_docs = get_cum_docs_per_zip(zips)  # cumulative num. of docs in zip files

    print('Sample word sequences from XMLs...')
    all_docs = get_docs(cum_docs, zips)
    all_words = [get_doc_words(doc) for doc in all_docs]
    all_seqs = sample_sequences(all_words, max_width=100)
    with open(os.path.join(PROJ_DIR, 'dl20', 'test_sequences.txt'), 'w') as f:
        for s in all_seqs:
            f.write('\t'.join(s) + '\n')
    print('Words sampled!')
