
import os
import random

import torch
import torch.nn.functional as F
import nltk
from allennlp.commands.elmo import ElmoEmbedder
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer, BertConfig, AutoConfig

from vars import PROJ_DIR, DEVICE


class Encoder(object):

    def __init__(self, params):

        emb_pars = {par.split('=')[0]: par.split('=')[1] for par in params.emb_pars}
        self.encoder = emb_pars['enc']
        self.elmo_dim = int(emb_pars['dim']) if 'dim' in emb_pars else None

        self.in_width = int(params.input_shape.split('x')[1])          # max. number of words from a doc
        self.enc_name = self.encoder[:4] if self.encoder[:4] == 'elmo' or self.encoder[:4] == 'bert' else self.encoder
        model_path = os.path.join(PROJ_DIR, 'models', self.enc_name)

        if self.enc_name == 'elmo':
            opt_fname = self.encoder + '_options.json'
            w_fname = '_'.join([self.encoder, 'weights.hdf5'])
            options_file = os.path.join(model_path, opt_fname)
            weight_file = os.path.join(model_path, w_fname)

            cuda_device = 0 if torch.cuda.is_available() else -1
            self.model = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)

        elif self.enc_name == 'bert':
            cache_dir = os.path.join(PROJ_DIR, 'models', 'bert')
            self.tokeniser = BertTokenizer.from_pretrained(self.encoder, cache_dir=cache_dir)
            self.model = BertModel.from_pretrained(self.encoder, cache_dir=cache_dir)
            self.model.eval()
            self.model.to(DEVICE)

        # TODO: add word2vec, Glove
        elif self.encoder == 'word2vec':
            pass
        elif self.encoder == 'glove':
            pass

    def sample_sequences(self, word_batch):

        seqs = []
        for wordlist in word_batch:
            diff = len(wordlist) - self.in_width
            if diff > 0:
                start = random.randint(0, diff)
                end = start + self.in_width
                seqs += [wordlist[start:end]]
            else:
                seqs += [wordlist]
        return seqs

    def encode_batch(self, seqs):

        if self.enc_name == 'elmo':
            embs = [torch.tensor(self.model.embed_sentence(s)) for s in seqs]
            embs = [emb[self.elmo_dim, ...] for emb in embs]

        elif self.enc_name == 'bert':
            inds = [torch.tensor(self.tokeniser.encode(s), device=DEVICE).unsqueeze(0) for s in seqs]
            with torch.no_grad():
                outputs = [self.model(i) for i in inds]
            embs = [tup[0].squeeze() for tup in outputs]

        elif self.enc_name == 'word2vec':
            # TODO: handle word2vec, glove et al.
            embs = None
        else:   # random

            embs = [torch.randn(len(s)) for s in seqs]

        embs = self.concat_embs(embs)
        # add channel dimension, and transpose last two dims for CNNs
        embs = torch.unsqueeze(embs, 1)
        embs = torch.transpose(embs, dim0=2, dim1=3)

        embs = embs.to(device=DEVICE)
        embs.requires_grad = True

        return embs

    def concat_embs(self, emb_list):

        embs = []
        for e in emb_list:
            diff = self.in_width - e.shape[0]
            if diff > 0:
                lp = int(diff / 2) if diff % 2 == 0 else int(diff // 2)
                rp = int(diff - lp)
                embs += [F.pad(e, [0, 0, lp, rp], 'constant', 0)] # pad from last to first dim
            else:
                embs += [e]

        return torch.stack(embs)