#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('true_file', type=str, help='Ground truth file in numpy.savetxt() format')
parser.add_argument('pred_file', type=str, help='Predictions file(s) in numpy.savetxt() format', nargs='+')
args = parser.parse_args()

y_true = np.loadtxt(args.true_file)
f1s = []

for pred_file in args.pred_file:
    y_pred = np.loadtxt(pred_file)
    assert(y_true.shape == y_pred.shape)
                        
    # f1 = f1_score(y_true, y_pred, average='micro')
    (prec, recall, f1, support) = precision_recall_fscore_support(y_true, y_pred, average='micro')

    print("{}: f1={}, prec={}, recall={}".format(pred_file, f1, prec, recall))
    f1s.append(f1)

if len(f1s) > 1:
    print("mean of f1s", np.mean(f1s))
