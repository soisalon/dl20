

import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data import get_model_savepath, DocDataset, SeqDataset
from vars import LOSSES, OPTIMS, MODEL_DIR, TESTING, DEVICE, PROJ_DIR
import cnn

parser = argparse.ArgumentParser()

# general params
parser.add_argument('--tr_ratio', nargs='?', type=float)
parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.1)
parser.add_argument('--seed', nargs='?', type=int, default=100)
parser.add_argument('--final', nargs='?', type=bool, default=False)  # whether to train with whole dataset
parser.add_argument('--use_seqs', nargs='?', type=bool, default=False)
# params for sampling and encoding words from XMLs
parser.add_argument('--emb_pars', nargs='*', default=['enc=elmo_2x1024_128_2048cnn_1xhighway', 'dim=2'])
# parser.add_argument('--emb_pars', nargs='*', default=['enc=bert-base-uncased'])
# parser.add_argument('--emb_pars', nargs='*', default=['enc=glove'])
# training params
parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
parser.add_argument('--batch_size', nargs='?', type=int, default=64)
parser.add_argument('--loss_fn', nargs='?', default='bce')
parser.add_argument('--optim', nargs='?', default='adadelta')
parser.add_argument('--opt_params', nargs='*', default=['lr=1.0'])
# CNN params
parser.add_argument('--model_name', nargs='?', default='DocCNN')          # BaseCNN / DocCNN
parser.add_argument('--n_conv_layers', nargs='?', type=int, default=2)
parser.add_argument('--kernel_shapes', nargs='*', default=['150x10', '2x2'])
parser.add_argument('--strides', nargs='*', default=['1x1', '1x1'])
parser.add_argument('--pool_sizes', nargs='*', default=['1x9', '1x5'])
parser.add_argument('--input_shape', nargs='?', default='256x100')
parser.add_argument('--n_kernels', nargs='*', type=int, default=[10, 10])
parser.add_argument('--conv_act_fn', nargs='?', default='relu')
parser.add_argument('--h_units', nargs='*', type=int, default=[64])
parser.add_argument('--fc_act_fn', nargs='?', default='relu')
parser.add_argument('--out_act_fn', nargs='?', default='sigmoid')
parser.add_argument('--dropout', nargs='?', type=float, default=0.5)

params = parser.parse_args()

print('params.use_seqs: ', params.use_seqs)

torch.manual_seed(params.seed)

n_classes = 126
n_docs = 299773                                 # docs (xml files) in total
n_docs_test = 33142
n_dev_docs = int(n_docs * params.dev_ratio)     # docs to use for dev set
n_tr_docs = n_docs - n_dev_docs                 # docs to use for training set

if params.tr_ratio:
    n_tr_docs = int(n_tr_docs * params.tr_ratio)

enc_name = params.emb_pars[0].split('=')[1]
enc_name = enc_name[:4] if enc_name[:4] == 'bert' or enc_name[:4] == 'elmo' else enc_name
emb_data_dir = os.path.join(PROJ_DIR, 'dl20', enc_name + '_data')

if TESTING:
    n_docs, n_tr_docs, n_dev_docs = 20, 12, 8
    params.batch_size = 4
    params.n_epochs = 2
    print('Testing code')


def train(epoch):
    model.train()
    for bi, (data, target) in enumerate(tr_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        opt.zero_grad()

        preds = model(data).double()
        loss = loss_fn(preds, target)
        loss.backward()
        opt.step()

        if bi % 100 == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, params.n_epochs, bi * len(data), len(tr_loader.dataset), 100. * bi / len(tr_loader), float(loss)))


def validate(lossv, pv, rv, fv):
    model.eval()
    val_loss, ps, rs, fs = 0, 0, 0, 0
    for bi, (data, target) in enumerate(dev_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        target = target.squeeze()
        with torch.no_grad():
            output = model(data).double()
        output = output.squeeze()

        val_loss += loss_fn(output, target).data.item()
        target = target.cpu().numpy()
        preds = (output >= 0.5).int().cpu().numpy()  # get the index of the max log-probability
        p, r, f, _ = precision_recall_fscore_support(target, preds, average='micro')
        ps += p
        rs += r
        fs += f

        if bi == 10:
            print('Predicions for inds {}-{} in sequences.txt.'.format(n_tr_docs + params.batch_size * 100,
                                                                       n_tr_docs + params.batch_size * 101))
            np.savetxt(os.path.join(PROJ_DIR, 'dl20', 'eyeball_preds.txt'), preds, fmt='%i')
            np.savetxt(os.path.join(PROJ_DIR, 'dl20', 'eyeball_targt.txt'), target, fmt='%i')
            # TODO: write topics predicted + true topics of this batch

    val_loss /= len(dev_loader)
    ps /= len(dev_loader)
    rs /= len(dev_loader)
    fs /= len(dev_loader)

    lossv += [val_loss]
    pv += [ps]
    rv += [rs]
    fv += [fs]

    ps = 100. * ps
    rs = 100. * rs
    fs = 100. * fs

    print('For model {}:'.format(model_fname))
    print('\nDev set: Average loss: {:.4f}, Precision: {:.0f}%, Recall: {}%, F1: {}%\n'.format(
        val_loss, ps, rs, fs))


# initialise CNN
model = getattr(cnn, params.model_name)
model = model(params=params)                  # init. model
model = model.to(DEVICE)                      # make sure model is set to correct device

loss_fn = LOSSES[params.loss_fn]()                              # get loss function
if params.opt_params[0] == 'default':
    opt = OPTIMS[params.optim](model.parameters())              # use default params if opt_params not given
else:                                                           # or get optimiser with given params
    opt_params = {par.split('=')[0]: float(par.split('=')[1]) for par in params.opt_params}
    opt = OPTIMS[params.optim](model.parameters(), **opt_params)

# load datasets
print('Initialise Datasets...')
seq_fpath = os.path.join(PROJ_DIR, 'dl20', 'sequences.txt')
te_seq_fpath = os.path.join(PROJ_DIR, 'dl20', 'test_sequences.txt')
if params.use_seqs:
    tr_dset = SeqDataset(seq_fpath, params, train=True) if not params.final \
        else SeqDataset(seq_fpath, params, train=True)
    dev_dset = SeqDataset(seq_fpath, params, train=False) if not params.final \
        else SeqDataset(te_seq_fpath, params, train=False)
else:
    tr_dset = DocDataset(enc_name + '_data', params, train=True)
    dev_dset = DocDataset(enc_name + '_data', params, train=False)

print('Done.')

print('Initialise DataLoaders...')
tr_loader = DataLoader(dataset=tr_dset, batch_size=params.batch_size, shuffle=True, num_workers=10)
dev_loader = DataLoader(dataset=dev_dset, batch_size=params.batch_size, shuffle=False, num_workers=10)
print('After init, torch.utils.data.get_worker_info(): ', torch.utils.data.get_worker_info())
print('Done.')

if DEVICE == torch.device('cuda'):
    print('mem allocated / reserved after starting dataloaders: ')
    print(torch.cuda.memory_allocated(device=DEVICE))
    print(torch.cuda.memory_reserved(device=DEVICE))
    print(torch.cuda.memory_summary(device=DEVICE))
    torch.cuda.empty_cache()

# get model path for saving
model_fname = get_model_savepath(params, ext='.pt')
model_path = os.path.join(MODEL_DIR, model_fname)  # path where trained model is saved

print('Start training...')
# train model
losses, precs, recs, fs = [], [], [], []
if not params.final:
    for e in range(params.n_epochs):
        train(e)
        validate(losses, precs, recs, fs)

else:
    for e in range(params.n_epochs):
        train(e)

    # get predictions on final test data
    test_data = dev_dset.dev_data
    with torch.no_grad():
        test_preds = model(test_data).squeeze()
    test_preds = test_preds.data.cpu().numpy()
    test_preds = (test_preds >= 0.5).astype('int')
    np.savetxt(os.path.join(PROJ_DIR, 'dl20', 'test_preds.txt'), test_preds, fmt='%i')

    torch.save(model.state_dict(), model_path)

# write some resulst into file
with open(os.path.join(PROJ_DIR, 'dl20', 'scores.txt'), 'a') as f:
    f.write('Scores for model after training for {} epochs: {}\n'.format(params.n_epochs, model_fname))
    f.write('Model precision = {}\n'.format(precs[-1]))
    f.write('Model recall = {}\n'.format(recs[-1]))
    f.write('Model F1= {}\n'.format(fs[-1]))
    f.write('\n#####\n')

