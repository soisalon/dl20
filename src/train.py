

import os
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data import get_eyeball_set, get_model_savepath, DocDataset, SeqDataset
from vars import LOSSES, OPTIMS, MODEL_DIR, TESTING, DEVICE, PROJ_DIR
import cnn

parser = argparse.ArgumentParser()

# general params
parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.1)
parser.add_argument('--seed', nargs='?', type=int, default=100)
parser.add_argument('--final', nargs='?', type=bool, default=False)  # whether to train with whole dataset
parser.add_argument('--use_seqs', nargs='?', type=bool, default=True)
parser.add_argument('--plot', nargs='?', type=bool, default=False)
# params for sampling and encoding words from XMLs
# parser.add_argument('--emb_pars', nargs='*', default=['enc=elmo_2x1024_128_2048cnn_1xhighway', 'dim=2'])
parser.add_argument('--emb_pars', nargs='*', default=['enc=random'])
# parser.add_argument('--emb_pars', nargs='*', default=['enc=glove'])
# training params
parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
parser.add_argument('--batch_size', nargs='?', type=int, default=64)
parser.add_argument('--loss_fn', nargs='?', default='bce')
parser.add_argument('--optim', nargs='?', default='adadelta')
parser.add_argument('--opt_params', nargs='*', default=['lr=1.0'])
parser.add_argument('--early_stop', nargs='?', default='F1')    # criterion for early stopping (F1 / loss)
# CNN params
parser.add_argument('--model_name', nargs='?', default='DocCNN')          # BaseCNN / DocCNN
parser.add_argument('--n_conv_layers', nargs='?', type=int, default=2)
parser.add_argument('--kernel_shapes', nargs='*', default=['150x10', '2x2'])
parser.add_argument('--strides', nargs='*', default=['1x1', '1x1'])
parser.add_argument('--pool_sizes', nargs='*', default=['1x9', '1x5'])
parser.add_argument('--input_shape', nargs='?', default='300x100')
parser.add_argument('--n_kernels', nargs='*', type=int, default=[10, 10])
parser.add_argument('--conv_act_fn', nargs='?', default='relu')
parser.add_argument('--h_units', nargs='*', type=int, default=[64])
parser.add_argument('--fc_act_fn', nargs='?', default='relu')
parser.add_argument('--out_act_fn', nargs='?', default='sigmoid')
parser.add_argument('--dropout', nargs='?', type=float, default=0.5)

params = parser.parse_args()

print('params.use_seqs: ', params.use_seqs)

torch.manual_seed(params.seed)

n_classes = 126                                 # number of different topics
n_docs = 299773                                 # docs (xml files) in total
n_docs_test = 33142                             # docs in test set
n_dev_docs = int(n_docs * params.dev_ratio)     # docs to use for dev set
n_tr_docs = n_docs - n_dev_docs                 # docs to use for training set

enc_name = params.emb_pars[0].split('=')[1]
enc_name = enc_name[:4] if enc_name[:4] == 'bert' or enc_name[:4] == 'elmo' else enc_name


def train(epoch, iter_count):
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
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tTraining loss for batch {}/{}: {:.6f}'.format(
                epoch + 1, params.n_epochs, bi * len(data), len(tr_loader.dataset), 100. * bi / len(tr_loader),
                bi + 1, len(tr_loader), float(loss)))

        # if plotting, get validation scores after every 1000 iterations
        if params.plot and bi % 1000 == 999:
            validate(n_its, losses, precs, recs, fscores, accs)
        iter_count += 1
    return iter_count


def validate(n_iters, lossv, pv, rv, fv, accv):
    n_iters += [it_count]
    model.eval()
    val_loss, corrs, ps, rs, fs = 0, 0, 0, 0, 0
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
        corrs += np.sum(preds == target)

        # if last epoch, get eyeball set from last full batch
        if bi == len(dev_loader) - 2 and (e == params.n_epochs - 1 or early_stop):
            # assuming dev set begins at index n_tr_docs
            st_i, end_i = n_tr_docs + params.batch_size * bi, n_tr_docs + params.batch_size * (bi + 1)
            seq_inds = [i for i in range(st_i, end_i)]
            get_eyeball_set(seq_inds, preds, target)
            print('After epoch {}/{}, for batch {}/{} - predictions for sequences {}-{} stored in eb_preds.txt.'
                  .format(e + 1, params.n_epochs, bi + 1, len(dev_loader), st_i, end_i))

    val_loss /= len(dev_loader)
    ps /= len(dev_loader)
    rs /= len(dev_loader)
    fs /= len(dev_loader)
    acc = corrs / (len(dev_loader.dataset) * n_classes)

    lossv += [val_loss]
    pv += [ps]
    rv += [rs]
    fv += [fs]
    accv += [acc]

    ps = 100. * ps
    rs = 100. * rs
    fs = 100. * fs
    acc = 100. * acc

    print('\nFor model {}:'.format(model_fname))
    print('Dev set - Average loss: {:.4f}, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%, Acc: {:.2f}%\n'.format(
        val_loss, ps, rs, fs, acc))

    return n_iters, lossv, pv, rv, fv, accv


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
tr_loader = DataLoader(dataset=tr_dset, batch_size=params.batch_size, shuffle=True, num_workers=10, drop_last=False)
dev_loader = DataLoader(dataset=dev_dset, batch_size=params.batch_size, shuffle=False, num_workers=10, drop_last=False)
print('After init, torch.utils.data.get_worker_info(): ', torch.utils.data.get_worker_info())
print('Done.')

# get model path for saving
model_fname = get_model_savepath(params, ext='.pt')
model_path = os.path.join(MODEL_DIR, model_fname)  # path where trained model is saved

print('Start training...')
# train model
losses, precs, recs, fscores, accs = [], [], [], [], []
n_its = []            # for plotting
it_count = 0
if not params.final:
    early_stop = False
    for e in range(params.n_epochs):
        # early stopping if F1 score has decreased / loss increased for two consecutive epochs, but train for one more
        if (params.early_stop == 'loss' and len(losses) > 10 and losses[-1] > losses[-2] > losses[-3]) or \
        (params.early_stop == 'F1' and len(fscores) > 10 and fscores[-1] < fscores[-2] < fscores[-3]):
            early_stop = True
            print('One more epoch before early stopping...')

        it_count = train(e, it_count)
        n_its, losses, precs, recs, fscores, accs = validate(n_its, losses, precs, recs, fscores, accs)

        if early_stop:
            break

else:
    for e in range(params.n_epochs):
        train(e, it_count)
        # no validation since training with the whole dataset

    # get predictions on final test data
    test_data = dev_dset.dev_data
    with torch.no_grad():
        test_preds = model(test_data).squeeze()
    test_preds = test_preds.data.cpu().numpy()
    test_preds = (test_preds >= 0.5).astype('int')
    np.savetxt(os.path.join(PROJ_DIR, 'dl20', 'test_preds.txt'), test_preds, fmt='%i')

    torch.save(model.state_dict(), model_path)

# write some results into file
with open(os.path.join(PROJ_DIR, 'dl20', 'scores.txt'), 'a') as f:
    # amount of actual epochs, in case of early stopping
    f.write('\nAfter training for {}/{} epochs, scores for model {}:\n'.format(e + 1, params.n_epochs, model_fname))
    f.write('Losses: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in losses])))
    f.write('Precisions: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in precs])))
    f.write('Recalls: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in recs])))
    f.write('F1-s: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fscores])))
    f.write('Accuracies: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in accs])))
    f.write('\n#####\n')

if params.plot:

    # write losses to file for plotting
    losses_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'final_losses.dat')
    precs_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'final_precs.dat')
    recs_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'final_recs.dat')
    fs_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'final_fs.dat')
    acc_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'final_accs.dat')
    iters_file = os.path.join(PROJ_DIR, 'dl20', 'plots', 'iters.dat')
    line = '{} layer {}\t' if params.n_conv_layers == 1 else '{} layers {}\t'
    with open(iters_file, 'w') as f:
        f.write('{}'.format('\t'.join(['{}'.format(it) for it in n_its])))
    with open(losses_file, 'a') as f:
        f.write(line.format(params.n_conv_layers, model_fname) +
                '\t'.join(['{:2.3f}'.format(s) for s in losses]) + '\n')
    with open(precs_file, 'a') as f:
        f.write(line.format(params.n_conv_layers, model_fname) +
                '\t'.join(['{:2.3f}'.format(s) for s in precs]) + '\n')
    with open(recs_file, 'a') as f:
        f.write(line.format(params.n_conv_layers, model_fname) +
                '\t'.join(['{:2.3f}'.format(s) for s in recs]) + '\n')
    with open(fs_file, 'a') as f:
        f.write(line.format(params.n_conv_layers, model_fname) +
                '\t'.join(['{:2.3f}'.format(s) for s in fscores]) + '\n')
    with open(acc_file, 'a') as f:
        f.write(line.format(params.n_conv_layers, model_fname) +
                '\t'.join(['{:2.3f}'.format(s) for s in accs]) + '\n')
