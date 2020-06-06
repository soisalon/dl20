
import os

import torch
import torch.nn.functional as F

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

from allennlp.commands.elmo import ElmoEmbedder
from transformers import BertModel, BertTokenizer

from vars import PROJ_DIR, DEVICE, EMB_DIR, TESTING


class Encoder(object):

    def __init__(self, params):

        emb_pars = {par.split('=')[0]: par.split('=')[1] for par in params.emb_pars}
        self.encoder = emb_pars['enc']
        self.elmo_dim = int(emb_pars['dim']) if 'dim' in emb_pars else None

        self.in_height, self.in_width = tuple(map(int, params.input_shape.split('x')))
        self.enc_name = self.encoder[:4] if self.encoder[:4] == 'elmo' or self.encoder[:4] == 'bert' else self.encoder
        model_path = os.path.join(PROJ_DIR, 'models', self.enc_name)

        if self.enc_name == 'elmo':
            opt_fname = self.encoder + '_options.json'
            w_fname = '_'.join([self.encoder, 'weights.hdf5'])
            options_file = os.path.join(model_path, opt_fname)
            weight_file = os.path.join(model_path, w_fname)

            # cuda_device = 0 if torch.cuda.is_available() else -1
            cuda_device = -1        # to avoid re-initialisation in subprocess
            self.model = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)

        elif self.enc_name == 'bert':
            cache_dir = os.path.join(PROJ_DIR, 'models', 'bert')
            self.tokeniser = BertTokenizer.from_pretrained(self.encoder, cache_dir=cache_dir)
            self.model = BertModel.from_pretrained(self.encoder, cache_dir=cache_dir)
            self.model.eval()
            # self.model.to(DEVICE) - avoid re-initialising CUDA

        elif self.encoder == 'word2vec':
            self.model = get_w2v_embs()
        elif self.encoder == 'glove':
            self.model = get_glove_embs()

    def encode_batch(self, seqs):

        if self.enc_name == 'elmo':
            embs = [torch.tensor(self.model.embed_sentence(s)) for s in seqs]
            embs = [emb[self.elmo_dim, ...] for emb in embs]

        elif self.enc_name == 'bert':
            inds = [torch.tensor(self.tokeniser.encode(' '.join(s)), device=DEVICE).unsqueeze(0) for s in seqs]
            with torch.no_grad():
                outputs = [self.model(i) for i in inds]
            embs = [tup[0].squeeze() for tup in outputs]

        elif self.enc_name == 'glove' or self.enc_name == 'word2vec':
            embs = []
            for s in seqs:
                e = torch.empty(len(s), self.in_height)
                for i, w in enumerate(s):
                    e[i, :] = torch.tensor(self.model[w]) if w in self.model else torch.randn(self.in_height)
                embs += [e]

        else:   # random
            embs = [torch.randn(self.in_height, len(s)) for s in seqs]

        embs = self.concat_embs(embs)
        # add channel dimension, and transpose last two dims for CNNs
        embs = torch.unsqueeze(embs, 1)
        embs = torch.transpose(embs, dim0=2, dim1=3)

        return embs

    def encode_seq(self, seq):

        emb = torch.zeros(self.in_width, self.in_height)

        if self.enc_name == 'elmo':
            e = torch.tensor(self.model.embed_sentence(seq))
            e = emb[self.elmo_dim, ...]

        elif self.enc_name == 'bert':
            # inds = torch.tensor(self.tokeniser.encode(' '.join(seq)), device=DEVICE).unsqueeze(0)
            inds = torch.tensor(self.tokeniser.encode(' '.join(seq))).unsqueeze(0)
            with torch.no_grad():
                output = self.model(inds)
            e = output[0].squeeze()

        elif self.enc_name == 'glove' or self.enc_name == 'word2vec':

            e = torch.empty(len(seq), self.in_height)
            for i, w in enumerate(seq):
                vec = self.model[w] if w in self.model else torch.empty(self.in_height).uniform_(-.25, .25)
                vec.flags.writeable = True      # to avoid user warning
                e[i, :] = torch.tensor(vec)

        else:   # random - sample from uniform dis. such that variance approx. the same as for w2v
            e = torch.empty(len(seq), self.in_height).uniform_(-.25, .25)

        elen = e.shape[0]
        diff = self.in_width - elen
        if diff > 0:
            lp = int(diff // 2)
            rp = int(diff - lp)
            emb[lp:self.in_width - rp] = e
        elif diff < 0:
            emb = e[:self.in_width]
        else:
            emb = e
        # add channel dimension, and transpose last two dims for CNNs
        emb = emb.T
        emb = torch.unsqueeze(emb, 0)
        return emb

    def concat_embs(self, emb_list):

        embs = torch.zeros(len(emb_list), self.in_width, self.in_height)
        for i, e in enumerate(emb_list):
            diff = self.in_width - e.shape[0]
            if diff > 0:
                lp = int(diff // 2)
                rp = int(diff - lp)
                # embs += [F.pad(e, [0, 0, lp, rp], 'constant', 0)]       # pad from last to first dim
                embs[i, lp:self.in_width - rp, :] = e
            elif diff < 0:
                embs[i] = e[:self.in_width, :]
            else:
                embs[i] = e

        return embs


def get_glove_embs(vec_path=os.path.join(EMB_DIR, 'glove', 'vecs.txt')):
    print('Loading Glove embeddings...')
    if not os.path.exists(vec_path):
        glove_fname = 'glove.840B.300d.txt' if not TESTING else 'glove.6B.300d.txt'
        glove2word2vec(glove_input_file=os.path.join(EMB_DIR, 'glove', glove_fname),
                       word2vec_output_file=vec_path)

    glove_model = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    return glove_model


def get_w2v_embs(vec_path=os.path.join(EMB_DIR, 'word2vec', 'GoogleNews-vectors-negative300.bin')):
    print('Loading w2vembeddings...')
    w2v_model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
    return w2v_model


if __name__ == '__main__':

    pass

    """
    n_parts = 20
    pinds = [0] + [n_docs // n_parts for _ in range(n_parts)]
    diff = n_docs - sum(pinds)
    if diff > 0:
        pinds[-1] += diff
    assert n_docs == sum(pinds)
    pinds = [pinds[i + 1] + pinds[i] for i in range(n_parts)]
    assert pinds[-1] == n_docs
    d_seqs = {k + 1: [] for k in range(n_parts)}
    for p in range(n_parts):
        p_seqs = all_seqs[pinds[p]:pinds[p + 1]]
        p_embs = emb_encoder.encode_batch(p_seqs)
        fpath = os.path.join(emb_data_dir, str(p) + '.pt')
        torch.save(p_embs, fpath)
    """