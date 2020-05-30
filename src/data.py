

import os
import zipfile
import glob
import xmltodict
import nltk
import string

from vars import DATA_DIR


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


def get_docs(inds):

    pattern = os.path.join(DATA_DIR,'*.zip')
    zips = sorted(glob.glob(pattern))       # for reading input batches
    cum_docs = get_cum_docs_per_zip(zips)   # cumulative num. of docs in zip files

    batch_docs = []
    for i in inds:
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


def get_doc_words(xmlfile, filter='nonalph'):
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

    title_tokens = nltk.word_tokenize(title)
    headline_tokens = nltk.word_tokenize(headline)
    sents_tokens = [nltk.word_tokenize(s) for s in sents if s]
    sents_tokens = [w for s in sents_tokens for w in s]

    words = title_tokens + headline_tokens + sents_tokens
    if filter == 'punct':
        words = [w for w in words if w not in string.punctuation]
    elif filter == 'nonalph':
        words = [w for w in words if w.isalpha()]        # filter out numeric words, they don't predict topic

    words = [w.lower() for w in words]
    words = [w for w in words if not len(w) < 4]        # filter words with length < 4
    if not words:
        print('No words in xmlfile:  ', xmlfile)
        print('doc_dict: ', doc_dict)

    return words


