

import os
import argparse
import random

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data import sample_sequences, get_model_savepath
from vars import LOSSES, OPTIMS, MODEL_DIR, TESTING, DEVICE, PROJ_DIR
from encoder import Encoder
import cnn


parser = argparse.ArgumentParser()

# general params
parser.add_argument('--tr_ratio', nargs='?', type=float)
parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.1)
parser.add_argument('--seed', nargs='?', type=int, default=100)
parser.add_argument('--cv_folds', nargs='?', type=int, default=1)
# params for sampling and encoding words from XMLs
parser.add_argument('--word_filter', nargs='?', default='nonalph')
# parser.add_argument('--emb_pars', nargs='*', default=['enc=elmo_2x1024_128_2048cnn_1xhighway', 'dim=2'])
parser.add_argument('--emb_pars', nargs='*', default=['enc=bert-base-uncased'])
# parser.add_argument('--emb_pars', nargs='*', default=['enc=word2vec'])
# training params
parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
parser.add_argument('--batch_size', nargs='?', type=int, default=32)
parser.add_argument('--loss_fn', nargs='?', default='bce')
parser.add_argument('--optim', nargs='?', default='adadelta')
parser.add_argument('--opt_params', nargs='*', default=['lr=1.0'])
# CNN params
parser.add_argument('--model_name', nargs='?', default='BaseCNN')          # BaseCNN / DocCNN
parser.add_argument('--n_conv_layers', nargs='?', type=int, default=1)
parser.add_argument('--kernel_shapes', nargs='*', default=['768x4', '1x2'])
parser.add_argument('--strides', nargs='*', default=['1x1'])
parser.add_argument('--pool_sizes', nargs='*', default=['1x2'])
parser.add_argument('--input_shape', nargs='?', default='768x100')


parser.add_argument('--n_kernels', nargs='*', type=int, default=[10])
parser.add_argument('--conv_act_fn', nargs='?', default='relu')
parser.add_argument('--h_units', nargs='*', type=int, default=[64])
parser.add_argument('--fc_act_fn', nargs='?', default='relu')
parser.add_argument('--out_act_fn', nargs='?', default='sigmoid')
parser.add_argument('--dropout', nargs='?', type=float, default=0.5)

params = parser.parse_args()

torch.manual_seed(params.seed)

if DEVICE == torch.device('cuda'):
    print('Cuda mem. stats:')
    print(torch.cuda.memory_summary(device=DEVICE))
    print('mem allocated: ')
    print(torch.cuda.memory_allocated(device=DEVICE))

n_classes = 126
n_docs = 299773                                 # docs (xml files) in total
n_dev_docs = int(n_docs * params.dev_ratio)     # docs to use for dev set
n_tr_docs = n_docs - n_dev_docs                 # docs to use for training set

if params.tr_ratio:
    n_tr_docs = int(n_tr_docs * params.tr_ratio)

in_height, in_width = tuple(map(int, params.input_shape.split('x')))        # desired width of input
kh = int(params.kernel_shapes[0].split('x')[0])
assert kh == in_height

labels = np.loadtxt(os.path.join(PROJ_DIR, 'dl20', 'ground_truth.txt'))
# labels = torch.tensor(labels, device=DEVICE, dtype=torch.float32)


if TESTING:
    n_docs, n_tr_docs, n_dev_docs = 20, 12, 8
    params.batch_size = 4
    params.n_epochs = 2
    print('Testing code')

print('Initialise embedding encoder')
emb_encoder = Encoder(params=params)

# get sequences corresponding to newsitems
with open(os.path.join(PROJ_DIR, 'dl20', 'sequences.txt'), 'r') as f:
    lines = [line.strip() for line in f]
    all_seqs = [line.split() for line in lines]
print('Seqs read from file')
all_seqs = sample_sequences(all_seqs, max_width=in_width)


# TODO: get all embs into 20 files, which are used by a DataLoader
enc_name = params.emb_pars[0].split('=')[1]
enc_name = enc_name[:4] if enc_name[:4] == 'bert' or enc_name[:4] == 'elmo' else enc_name
emb_data_dir = os.path.join(PROJ_DIR, 'dl20', enc_name + '_data')
fpath = os.path.join(emb_data_dir, 'all.pt')
if not os.path.exists(emb_data_dir):
    os.makedirs(emb_data_dir)
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
    p_embs = emb_encoder.encode_batch(all_seqs)
    torch.save(p_embs, fpath)

else:
    all_embs = torch.load(fpath)

# all_embs = emb_encoder.encode_batch(all_seqs)
if DEVICE == torch.device('cuda'):
    print('mem allocated after loading labels and init. encoder: ')
    print(torch.cuda.memory_allocated(device=DEVICE))
    print(torch.cuda.memory_summary(device=DEVICE))


def train(mdl, input_inds, out_labels):
    # training loop
    n_iters = len(input_inds) // params.batch_size
    losses = []
    stop = False
    for epoch in range(params.n_epochs):
        for it in range(n_iters):
            batch_inds = input_inds[it * params.batch_size: (it + 1) * params.batch_size]

            seqs = [all_seqs[i] for i in batch_inds]
            batch = emb_encoder.encode_batch(seqs)  # get encoded batch
            batch.requires_grad = True

            opt.zero_grad()

            preds = mdl(batch)
            preds = preds.squeeze()
            actual = out_labels[it * params.batch_size: (it + 1) * params.batch_size]

            loss = loss_fn(preds, actual)

            loss.backward()
            opt.step()

            if it % 100 == 0:
                print('at iter {}, loss =  {}'.format(it, loss))

            losses += [float(loss)]
            # stop if loss is not changing
            if len(losses) > 100:
                if all(abs(losses[j - 1] - losses[j]) < 1e-1 for j in range(len(losses) - 1, len(losses) - 11, -1)):
                    stop = True
                    break

        # for other than the last epoch, print loss and acc for the dev set
        if epoch < params.n_epochs - 1:
            mdl.eval()
            val_preds = mdl(dev_embs).squeeze()
            val_loss = loss_fn(val_preds, dev_labels)
            val_preds = (val_preds >= 0.5).int()

            p, r, f, _ = precision_recall_fscore_support(dev_labels.numpy(), val_preds.numpy(), average='micro')

            # acc = torch.sum(val_preds == dev_labels.int()).float() / (n_dev_docs * 126)
            print('After epoch {}/{}\nDev loss = {}'.format(epoch + 1, params.n_epochs, float(val_loss)))
            print('Metrics: P - {}, R - {}, F1 - {}'.format(p, r, f))
            mdl.train()
        if stop:
            break
    return model


# get model path for saving
model_fname = get_model_savepath(params, ext='.pt')
model_path = os.path.join(MODEL_DIR, model_fname)       # path where trained model is saved

# initialise CNN
model = getattr(cnn, params.model_name)
model = model(params=params)                  # init. model
model = model.to(DEVICE)                      # make sure model is set to correct device

if DEVICE == torch.device('cuda'):
    print('mem allocated after initialising CNN: ')
    print(torch.cuda.memory_allocated(device=DEVICE))
    print(torch.cuda.memory_summary(device=DEVICE))

loss_fn = LOSSES[params.loss_fn]()                              # get loss function
if params.opt_params[0] == 'default':
    opt = OPTIMS[params.optim](model.parameters())              # use default params if opt_params not given
else:                                                           # or get optimiser with given params
    opt_params = {par.split('=')[0]: float(par.split('=')[1]) for par in params.opt_params}
    opt = OPTIMS[params.optim](model.parameters(), **opt_params)

all_inds = [i for i in range(n_docs)]
random.shuffle(all_inds)

print('Start training / cross-validation with {} fold(s)'.format(params.cv_folds))
accs = []             # for storing accuracies
rand_accs = []        # accs of random guesses
fs = []
precs = []
recs = []
for fold in range(params.cv_folds):

    print('Get tr and dev inds...')
    if params.cv_folds == 1:        # not doing CV, take random samples
        tr_inds = all_inds[:n_tr_docs]
        dev_inds = all_inds[n_tr_docs:]
    else:
        assert params.cv_folds * params.dev_ratio == 1.0
        dev_inds = [all_inds[i] for i in range(fold * n_dev_docs, (fold + 1) * n_dev_docs)]
        tr_inds = [i for i in all_inds if i not in dev_inds]

    print('Done.')
    # get training and dev. labels
    print('get tr, dev labels...')
    tr_labels = torch.tensor(np.take(labels, tr_inds, axis=0), device=DEVICE, dtype=torch.float32)
    dev_labels = torch.tensor(np.take(labels, dev_inds, axis=0), device=DEVICE, dtype=torch.float32)
    print('Done.')

    if DEVICE == torch.device('cuda'):
        print('mem allocated / reserved after setting labels to device: ')
        print(torch.cuda.memory_allocated(device=DEVICE))
        print(torch.cuda.memory_reserved(device=DEVICE))
        print(torch.cuda.memory_summary(device=DEVICE))
        torch.cuda.empty_cache()

    # for validation
    dev_seqs = [all_seqs[i] for i in dev_inds]
    dev_embs = emb_encoder.encode_batch(dev_seqs)

    if DEVICE == torch.device('cuda'):
        print('mem allocated / reserved after getting dev_embs: ')
        print(torch.cuda.memory_allocated(device=DEVICE))
        print(torch.cuda.memory_reserved(device=DEVICE))
        print(torch.cuda.memory_summary(device=DEVICE))

    # train model

    model = train(model, tr_inds, tr_labels)

    # get accuracy on dev set, having trianed with whole trainiing fold
    dev_preds = model(dev_embs).squeeze()
    dev_preds = (dev_preds >= 0.5).int()
    accs += [(torch.sum(dev_preds == dev_labels.int()).float() / (n_dev_docs * 126)).item()]

    # random choice
    rand_preds = (torch.rand(n_dev_docs, n_classes) >= 0.5).int()
    rand_accs += [(torch.sum(rand_preds == dev_labels.int()).float() / (n_dev_docs * 126)).item()]

    # other metrics
    p, r, f, _ = precision_recall_fscore_support(dev_labels.numpy(), dev_preds.numpy(), average='micro')
    precs += [p]
    recs += [r]
    fs += [f]

    # sample a few (strongly) misclassified newsitems
    bad_preds = (torch.sum((dev_preds == dev_labels.int()), dim=1).float() / 126 < 0.5).nonzero().squeeze()
    bad_preds = [p.item() for p in bad_preds]
    print('Examples of badly classified docs: {}'.format(
        [dev_inds[i] for i in random.sample(bad_preds, k=min(10, len(bad_preds)))]))

with open(os.path.join(PROJ_DIR, 'dl20', 'scores.txt'), 'w') as f:
    f.write('Scores for model: {}\n'.format(model_fname))
    for fld in range(params.cv_folds):
        f.write('In fold {}/{}:\n'.format(fld + 1, params.cv_folds))
        f.write('Model accuracy = {}\n'.format(accs[fld]))
        f.write('Model precision = {}\n'.format(precs[fld]))
        f.write('Model recall = {}\n'.format(recs[fld]))
        f.write('Model F1= {}\n'.format(fs[fld]))
        f.write('Random guess: {}\n'.format(rand_accs[fld]))
    f.write('\n#####\n')

print('Average model accuracy: ', np.mean(accs))
print('Avg. random guess: ', np.mean(rand_accs))

"""
# train final model on the whole training dataset, and save to file
final_model = getattr(cnn, params.model_name)(params=params)
final_model= final_model.to(DEVICE)

labels = torch.tensor(labels)
final_model = train(final_model, all_inds, labels)
torch.save(final_model.state_dict(), model_path)
"""

