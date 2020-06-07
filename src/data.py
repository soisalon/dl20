

import os
import zipfile
import glob
import random

import xmltodict
import nltk
import torch
import numpy as np

from vars import PROJ_DIR, TESTING, DEVICE, DATA_DIR
from encoder import Encoder


class SeqDataset(torch.utils.data.Dataset):
    """
    Dataset class where word sequences are encoded as in get_item.
    """
    def __init__(self, fpath, tr_inds, params, encoder, train=True):

        self.train = train
        self.use_whole = params.final

        self.encoder = encoder

        # n_docs_test = 33142
        # n_tr = n_docs - n_dev
        # n_dev = int(n_docs * params.dev_ratio)
        n_docs = 299773 if not TESTING else 1000

        dev_inds = [i for i in range(n_docs) if i not in tr_inds]

        with open(fpath, 'r') as f:
            lines = [l.strip() for l in f]
        seqs = [l.split() for l in lines]
        if TESTING:
            seqs = seqs[:100]

        labels_path = os.path.join(PROJ_DIR, 'dl20', 'ground_truth.txt')
        labels = np.loadtxt(labels_path)

        if self.train and not self.use_whole:
            self.tr_seqs = [seqs[i] for i in tr_inds]
            self.tr_labels = torch.tensor(np.take(labels, indices=tr_inds, axis=0))
        elif not self.train and not self.use_whole:
            self.dev_seqs = [seqs[i] for i in dev_inds]
            self.dev_labels = torch.tensor(np.take(labels, indices=dev_inds, axis=0))
        elif self.train and self.use_whole:
            self.tr_seqs = seqs
            self.tr_labels = labels
        else:
            self.dev_seqs = seqs

    def __getitem__(self, idx):
        if self.train:
            item, target = self.transform(self.tr_seqs[idx]), self.tr_labels[idx]
            return item, target
        elif not self.train and not self.use_whole:
            item, target = self.transform(self.dev_seqs[idx]), self.dev_labels[idx]
            return item, target
        else:
            item = self.transform(self.dev_seqs[idx])
            return item

    def __len__(self):
        if self.train:
            return len(self.tr_seqs)
        else:
            return len(self.dev_seqs)

    def transform(self, seq):
        return self.encoder.encode_seq(seq)


class DocDataset(torch.utils.data.Dataset):
    """
    Dataset which loads the whole dataset into a tensor.

    """
    def __init__(self, root_dir, params, train=True):

        self.train = train
        self.all = params.final

        in_height, in_width = tuple(map(int, params.input_shape.split('x')))
        n_docs = 299773
        n_docs_test = 33142
        n_dev = int(n_docs * params.dev_ratio)
        n_tr = n_docs - n_dev

        data_path = os.path.join(PROJ_DIR, 'dl20', root_dir)

        if not self.train and self.all: # test data
            data = torch.empty(n_docs_test, 1, in_height, in_width, device=DEVICE)
            n_parts = 10
        else:
            data = torch.zeros(n_docs, 1, in_height, in_width)
            n_parts = 100 if not TESTING else 10
        i = 0
        for fi in range(n_parts):
            fn = 'te_' + str(fi) + '.pt' if not self.train and self.all else str(fi) + '.pt'
            fp = os.path.join(data_path, fn)
            t = torch.load(fp)
            tlen = t.shape[0]
            data[i:i + tlen] = t
            i += tlen
        print('t.shape: ', t.shape)
        print('data.shape: ', data.shape)

        labels_path = os.path.join(PROJ_DIR, 'dl20', 'ground_truth.txt')
        labels = torch.tensor(np.loadtxt(labels_path))

        if self.train and not self.all:
            self.tr_data = data[:n_tr]
            self.tr_labels = labels[:n_tr]
        elif not self.train and not self.all:
            self.dev_data = data[n_tr:]
            self.dev_labels = labels[n_tr:]
        elif self.train and self.all:
            self.tr_data = data
            self.tr_labels = labels
        else:
            self.dev_data = data

    def preprocess(self):
        pass

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
    """
    Get the path to which a model will be saved, based on the parameters.
    :param params:
    :param ext:
    :return:
    """

    mod = 'Doc' if params.model_name == 'DocCNN' else 'Base'
    enc = params.emb_pars[0].split('=')[1]
    enc = 'w2v' if enc == 'word2vec' else enc[:4]
    encoder = 'enc=' + enc
    nl = 'nl' + str(params.n_conv_layers)
    ks = 'ks' + '+'.join(params.kernel_shapes)
    pls = 'ps' + '+'.join(params.pool_sizes) if mod == 'Doc' else ''
    sts = 'st' + '+'.join(params.strides) if mod == 'Doc' else ''
    dils = 'di' + '+'.join(params.dilations) if mod == 'Doc' else ''
    pads = 'pd' + '+'.join(params.paddings) if mod == 'Doc' else ''
    insh = 'in' + params.input_shape
    nk = 'nk' + '+'.join([str(n) for n in params.n_kernels])
    caf, faf, oaf = params.conv_act_fn[:3], params.fc_act_fn[:3], params.out_act_fn[:3]
    d = 'd' + str(params.dropout)

    bs, ne, op, ls = 'bs' + str(params.batch_size), 'ep' + str(params.n_epochs), params.optim[:3], params.loss_fn[:3]
    op_pars = '+'.join(params.opt_params) if params.opt_params != 'default' else 'def'

    hu = 'h' + '+'.join([str(n) for n in params.h_units])

    if mod == 'Doc':
        savepath = '-'.join([mod, encoder, nl, ks, sts, pls, dils, pads, insh, nk, caf, faf, oaf, d, hu, bs, ne, op, op_pars,
                            ls]) + ext
    else:
        savepath = '-'.join(
            [mod, encoder, nl, ks, insh, nk, caf, faf, oaf, d, hu, bs, ne, op, op_pars, ls]) + ext

    return savepath


def get_docs(cum_docs, zips):
    """
    Get the newsitems as list of XML documents.
    :param cum_docs:
    :param zips:
    :return:
    """
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
    """
    Count the cumulative number of XMLs in zip files.
    :param zips:
    :return:
    """
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
    """
    Take a random contiguous sequence from a list of words extracted from an XML document.
    :param word_batch:
    :param max_width:
    :return:
    """
    random.seed(100)
    seqs = []
    for wi, wordlist in enumerate(word_batch):
        diff = len(wordlist) - max_width
        if diff > 0:
            start = random.randint(0, diff)
            end = start + max_width
            seqs += [wordlist[start:end]]
        else:
            seqs += [wordlist]
        if wi % 100 == 0:
            print('{} seqs done!'.format(wi + 1))
    return seqs


def get_eyeball_set(seq_inds, preds, target, model_name):
    """
    Write to file some sequences, predictions and actual labels (for error analysis).
    :param seq_inds:
    :param preds:
    :param target:
    :param model_name:
    :return:
    """
    # write sequences and corresponding preds and target values in a file
    codes_fp = os.path.join(PROJ_DIR, 'dl20', 'codes', 'topic_codes.txt')
    eb_fp = os.path.join(PROJ_DIR, 'dl20', 'eyeball', 'eb_{}.txt'.format(model_name))
    seqs_fp = os.path.join(PROJ_DIR, 'dl20', 'sequences.txt')
    # get topics
    with open(codes_fp, 'r') as f:
        code_lines = [line.strip().split() for line in f if not line.startswith(';')]
    topics = [line[1] for line in code_lines]
    # get seqs
    with open(seqs_fp, 'r') as f:
        eb_seqs = [line.strip() for line in f]
    eb_seqs = [eb_seqs[i] for i in seq_inds]

    for i in range(len(seq_inds)):
        pred_topic_inds = np.where(preds[i] == 1)[0]
        tgt_topic_inds = np.where(target[i] == 1)[0]
        pred_topics = [topics[j] for j in pred_topic_inds]
        tgt_topics = [topics[j] for j in tgt_topic_inds]

    with open(eb_fp, 'w') as f:
        for si, s in zip(seq_inds, eb_seqs):
            f.write('Sequence {}. Predicted: {} - Actual: {}\n'.format(si, ', '.join(pred_topics),
                                                                       ', '.join(tgt_topics)))
            s = s.replace('\t', ' ')
            f.write('Words: {}\n'.format(s))
            f.write('\n######\n')


if __name__ == '__main__':

    # get words representing newsitems into a text file
    import argparse

    n_classes = 126
    n_docs = 299773  # docs (xml files) in total
    n_docs_test = 33142

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_pars', nargs='*', default=['enc=elmo_2x1024_128_2048cnn_1xhighway', 'dim=2'])
    parser.add_argument('--input_shape', nargs='?', default='256x100')
    parser.add_argument('--set', nargs='?', default='train')
    params = parser.parse_args()

    # sample sequences from zip files
    if params.set == 'seqs':
        pattern = os.path.join(DATA_DIR, '*.zip')
        zips = sorted(glob.glob(pattern))  # for reading input batches
        cum_docs = get_cum_docs_per_zip(zips)  # cumulative num. of docs in zip files

        print('Sample word sequences from XMLs...')
        all_docs = get_docs(cum_docs, zips)
        all_words = [get_doc_words(doc) for doc in all_docs]
        all_seqs = sample_sequences(all_words, max_width=100)
        with open(os.path.join(PROJ_DIR, 'dl20', 'sequences.txt'), 'w') as f:
            for s in all_seqs:
                f.write('\t'.join(s) + '\n')
        print('Words sampled!')

    emb_encoder = Encoder(params=params)

    # encode training sequences into tensors
    if params.set == 'train':
        fname= 'sequences.txt'
        N = n_docs
    # or encode test seqences into tensors
    else:
        fname = 'test_sequences.txt'
        N = n_docs_test

    seq_fpath = os.path.join(PROJ_DIR, 'dl20', fname)
    print('get sequences from file: ', fname)
    with open(seq_fpath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        seqs = [line.split() for line in lines]
    print('Seqs read from file.')
    print('len(seqs): ', len(seqs))
    print('seqs[-1]: ', seqs[-1])
    n_parts = 10 if params.set == 'test' else 100

    pinds = [0] + [N // n_parts for _ in range(n_parts)]
    pinds = [0] + [pinds[i] + sum(pinds[:i]) for i in range(1, len(pinds))]
    diff = N - pinds[-1]
    if diff > 0:
        pinds[-1] += diff
    assert N == pinds[-1]
    for p in range(n_parts):
        print('Get embs for range: {}--{}'.format(pinds[p], pinds[p + 1]))
        p_seqs = seqs[pinds[p]:pinds[p + 1]]
        p_embs = emb_encoder.encode_batch(p_seqs)
        dname = 'te_' + str(p) + '.pt' if params.set == 'test' else str(p) + '.pt'
        dpath = os.path.join(PROJ_DIR, 'dl20', emb_encoder.enc_name + '_data', dname)
        torch.save(p_embs, dpath)
        del p_embs
        print('Part {} (test) encoded!'.format(p))
