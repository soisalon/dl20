

import os
import torch

if os.getcwd().split(os.sep)[1] == 'wrk': # if working on Ukko2

    TESTING = True
else:
    TESTING = False

# make sure you are in the project dir (parent of source)
DATA_PATH = os.path.join(os.getcwd(), 'corpus')

LOSSES = {'mse': torch.nn.MSELoss, 'bce': torch.nn.BCELoss}
OPTIMS = {'sgd': torch.optim.SGD, 'adadelta': torch.optim.Adadelta, 'adam': torch.optim.Adam}

ACTIVATIONS = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid(), 'tanh': torch.nn.Tanh(),
               'softmax': torch.nn.Softmax()}