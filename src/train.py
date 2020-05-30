

import os
import argparse

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data import get_doc_words, get_docs, get_model_savepath
from vars import LOSSES, OPTIMS, MODEL_DIR, TESTING, DEVICE, PROJ_DIR
from encoder import Encoder
import cnn


parser = argparse.ArgumentParser()

# general params
parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.1)
parser.add_argument('--seed', nargs='?', type=int, default=100)
parser.add_argument('--cv_folds', nargs='?', type=int)
# params for sampling and encoding words from XMLs
parser.add_argument('--word_filter', nargs='?', default='nonalph')
parser.add_argument('--emb_pars', nargs='*', default=['enc=elmo_2x1024_128_2048cnn_1xhighway', 'dim=2'])
# training params
parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
parser.add_argument('--batch_size', nargs='?', type=int, default=32)
parser.add_argument('--loss_fn', nargs='?', default='bce')
parser.add_argument('--optim', nargs='?', default='adadelta')
parser.add_argument('--opt_params', nargs='*', default=['lr=1.0'])
# CNN params
parser.add_argument('--model_name', nargs='?', default='BaseCNN')          # BaseCNN / DocCNN
parser.add_argument('--n_conv_layers', nargs='?', type=int, default=1)
parser.add_argument('--kernel_shapes', nargs='*', default=['256x2', '1x2'])
parser.add_argument('--strides', nargs='*', default=['1x1'])
parser.add_argument('--pool_sizes', nargs='*', default=['1x2'])
parser.add_argument('--input_shape', nargs='?', default='256x50')
parser.add_argument('--n_kernels', nargs='*', type=int, default=[10])
parser.add_argument('--conv_act_fn', nargs='?', default='relu')
parser.add_argument('--h_units', nargs='*', type=int, default=[64])
parser.add_argument('--fc_act_fn', nargs='?', default='relu')
parser.add_argument('--out_act_fn', nargs='?', default='sigmoid')
parser.add_argument('--dropout', nargs='?', type=float, default=0.5)

params = parser.parse_args()

torch.manual_seed(params.seed)


n_classes = 126
n_docs = 299773                                 # docs (xml files) in total
n_dev_docs = int(n_docs * params.dev_ratio)     # docs to use for dev set
n_tr_docs = n_docs - n_dev_docs                 # docs to use for training set
if params.cv_folds:
    params.dev_ratio = 1 / params.cv_folds
    n_dev_docs = int(n_docs * params.dev_ratio)
    n_tr_docs = n_docs - n_dev_docs
else:
    params.cv_folds = 1

in_width = int(params.input_shape.split('x')[1])        # desired width of input

labels = np.loadtxt(os.path.join(PROJ_DIR, 'ground_truth.txt'))

if TESTING:
    m_docs = 20
    params.batch_size = 4
    params.n_epochs = 1


def train(mdl, input_inds, out_labels):
    # training loop
    n_iters = len(input_inds) // params.batch_size
    for epoch in range(params.n_epochs):
        for it in range(n_iters):
            batch_inds = tr_inds[it * params.batch_size: (it + 1) * params.batch_size]

            batch_docs = get_docs(batch_inds)  # get xml files corresponding to indices

            # get words from docs, filtered
            batch_words = [get_doc_words(doc, filter=params.word_filter) for doc in batch_docs]

            seqs = emb_encoder.sample_sequences(batch_words)  # get sequences of given length

            batch = emb_encoder.encode_batch(seqs)  # get encoded batch

            opt.zero_grad()

            preds = mdl(batch)
            preds = preds.squeeze()
            actual = out_labels[it * params.batch_size: (it + 1) * params.batch_size]

            loss = loss_fn(preds, actual)

            loss.backward()
            opt.step()

            if it % 500 == 0:
                print('at iter {}, loss =  {}'.format(it, loss))
    return model


# get model path for saving
model_fname = get_model_savepath(params, ext='.pt')
model_path = os.path.join(MODEL_DIR, model_fname)       # path where trained model is saved

model = getattr(cnn, params.model_name)
model = model(params=params)                  # init. model
model = model.to(DEVICE)                      # make sure model is set to correct device

loss_fn = LOSSES[params.loss_fn]()                              # get loss function
if params.opt_params == 'default':
    opt = OPTIMS[params.optim](model.parameters())              # use default params if opt_params not given
else:                                                           # or get optimiser with given params
    opt_params = {par.split('=')[0]: float(par.split('=')[1]) for par in params.opt_params}
    opt = OPTIMS[params.optim](model.parameters(), **opt_params)


emb_encoder = Encoder(params=params)            # for encoding words with embeddings

accs = torch.zeros(params.cv_folds)             # for storing accuracies
rand_accs = torch.zeros(params.cv_folds)        # accs of random guesses
fs = torch.zeros(params.cv_folds)
precs = torch.zeros(params.cv_folds)
recs = torch.zeros(params.cv_folds)
for fold in range(params.cv_folds):

    if params.cv_folds == 1:        # not doing CV, dev set from the end part of data
        tr_inds = [i for i in range(n_tr_docs)]
        dev_inds = [i for i in range(n_tr_docs, n_docs)]
    else:
        dev_inds = [i for i in range(fold * n_dev_docs, (fold + 1) * n_dev_docs)]
        tr_inds = [i for i in range(n_docs) if i not in dev_inds]

    # get training and dev. labels
    tr_labels = torch.tensor(np.take(labels, tr_inds, axis=0), device=DEVICE, dtype=torch.float32)
    dev_labels = torch.tensor(np.take(labels, dev_inds, axis=0), device=DEVICE, dtype=torch.float32)

    model = train(model, tr_inds, tr_labels)

    dev_docs = get_docs(dev_inds)
    dev_words = [get_doc_words(doc, filter=params.word_filter) for doc in dev_docs]
    dev_seqs = emb_encoder.sample_sequences(dev_words)
    dev_embs = emb_encoder.encode_batch(dev_seqs)
    dev_preds = model(dev_embs).squeeze()

    # get accuracy on dev set
    dev_preds = (dev_preds >= 0.5).int()
    accs[fold] = torch.sum(dev_preds == dev_labels.int()).float() / n_dev_docs

    rand_preds = (torch.rand(n_dev_docs, n_classes) >= 0.5).int()
    rand_accs[fold] = torch.sum(rand_preds == dev_labels.int()).float() / n_dev_docs

    precs[fold], recs[fold], fs[fold], _ = precision_recall_fscore_support(dev_labels, dev_preds, average='micro')


print('Scores for model: {}'.format(model_fname))
for f in range(params.cv_folds):
    print('In fold {}/{}: '.format(f + 1, params.cv_folds))
    print('Model accuracy = {}'.format(accs[f]))
    print('Random guess: ', rand_accs[f])

print('Average model accuracy: ', torch.mean(accs))
print('Avg. random guess: ', torch.mean(rand_accs))

# train final model on the whole training dataset, and save to file
final_model = getattr(cnn, params.model_name)(params=params)
final_model= final_model.to(DEVICE)
all_inds = [i for i in range(n_docs)]
labels = torch.tensor(labels)
final_model = train(final_model, all_inds, labels)
torch.save(final_model.state_dict(), model_path)


