

import os
import zipfile
import glob
import xmltodict
import nltk
import string
import collections


def embed_doc(words, width, embedding='elmo'):
    """
    Embed a document into a matrix of size (embedding_dim, width).

    :param words:
    :param width
    :param embedding:
    :return:
    """
    #TODO: pad

    pass

def get_cum_docs_per_zip(zips):

    docs_per_zip = {}
    count = 0
    for zipf in zips:
        with zipfile.ZipFile(zipf) as zf:
            count += len(zf.namelist())
            docs_per_zip[zipf] = count
    return docs_per_zip


def get_doc_words(xmlfile, filter=None):
    """
    Extract words from doc: <title>, <headline>, and <text> (<p> sadasd </p>), and put the woords into a list.
    :param xmlfile:
    :param filter:
    :return:
    """
    doc_dict = xmltodict.parse(xmlfile)

    keys1 = list(doc_dict.keys())
    if len(keys1) > 1:
        print('More than one key in root level: ', keys1)
    k1 = keys1[0]
    keys2 = [k2 for k1 in doc_dict for k2 in doc_dict[k1]]

    title = doc_dict[k1]['title'] if 'title' in keys2 else []
    headline = doc_dict[k1]['headline'] if 'headline' in keys2 else []
    text = doc_dict[k1]['text'] if 'text' in keys2 else []

    if type(text) != collections.OrderedDict:
        print('Type of <text> is not ordered dict but: ', type(text))
        print('in xmlfile: ', xmlfile)
        sents = []
    elif not doc_dict[k1]['text']['p']:
        sents = []
        print('no <p> in <text> for xmlfile: ', xmlfile)
        print('text: ', text)
    else:
        sents = doc_dict[k1]['text']['p']

    #if 'p' not in doc_dict[k1]['text']:
    #    print('p missing from text, print text: ', doc_dict[k1]['text'])
    #    sents = doc_dict[k1]['text']

    title_tokens = nltk.word_tokenize(title) if title else []
    headline_tokens = nltk.word_tokenize(headline) if headline else []
    try:
        sents_tokens = [nltk.word_tokenize(s) for s in sents if s]
    except TypeError:
        print('error with tkoenisation')

    sents_tokens = [w for s in sents_tokens for w in s] if text else []

    words = title_tokens + headline_tokens + sents_tokens
    if filter == 'punct':
        words = [w for w in words if w not in string.punctuation]
    elif filter == 'nonalph':
        words = [w for w in words if w.isalpha()]
    words = [w.lower() for w in words]
    if not words:
        print('No words in xmlfile:  ', xmlfile)
        print('doc_dict: ', doc_dict)

    return words


if __name__ == '__main__':

    # TODO: get the max. number of words in a doc

    print(os.getcwd())

    pattern = os.path.join('../corpus', '*.zip')
    n_words = []
    n_words_punct = []
    n_words_alph = []
    n_nones = 0
    d1_keys = []
    d2_keys = []
    d3_keys = []
    n_docs = 0
    for zipf in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(zipf, 'r') as zf:
            for xmlfile in zf.namelist():
                with zf.open(xmlfile, 'r') as xf:
                    doc = xf.read()
                    n_docs += 1
                    doc_dict = xmltodict.parse(doc)

                    if not doc_dict:
                        print('Dict None, when xmlfile = {} and zipf = {}'.format(xmlfile, zipf))
                        n_nones += 1
                    else:
                        d1_keys += [k for k in doc_dict if k not in d1_keys]
                        d2_keys += [k2 for k1 in doc_dict for k2 in doc_dict[k1]]
                        for k in doc_dict:
                            for k2 in doc_dict[k]:
                                if type(doc_dict[k][k2]) == collections.OrderedDict:
                                    d3_keys += list(doc_dict[k][k2].keys())

                        n_words += [len(get_doc_words(doc))]
                        n_words_punct += [len(get_doc_words(doc, filter='punct'))]
                        n_words_alph += [len(get_doc_words(doc, filter='nonalph'))]

    print('max(n_words): ', max(n_words))
    print('max(n_words_punct): ', max(n_words_punct))
    print('max(n_words_alph): ', max(n_words_alph))

    print('n_nones: ', n_nones)
    print('n_Docs: ', n_docs)

    print('len(d1_keys: ', len(d1_keys))
    print('len(d2_keys: ', len(d2_keys))
    print('len(d3_keys: ', len(d3_keys))

