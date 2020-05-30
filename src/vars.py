

import os
import torch

if os.getcwd().split(os.sep)[1] == 'wrk':   # if working on Ukko2
    TESTING = False
    PROJ_DIR = '/wrk/users/eliel/projects/dl_course20/'
else:
    TESTING = True
    PROJ_DIR = '/Users/eliel/Projects/courses/DL20/project/'

DATA_DIR = os.path.join(PROJ_DIR, 'dl20', 'corpus')
MODEL_DIR = os.path.join(PROJ_DIR, 'models', 'trained')
CACHE_DIR = os.path.join(PROJ_DIR, 'models', 'cache')
paths = (MODEL_DIR, CACHE_DIR)
for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

LOSSES = {'mse': torch.nn.MSELoss, 'bce': torch.nn.BCELoss, 'nll': torch.nn.NLLLoss, 'cre': torch.nn.CrossEntropyLoss}
OPTIMS = {'sgd': torch.optim.SGD, 'adadelta': torch.optim.Adadelta, 'adam': torch.optim.Adam}

ACTIVATIONS = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid(), 'tanh': torch.nn.Tanh(),
               'softmax': torch.nn.Softmax()}

EMB_DIMS = {'word2vec': 300, 'random': 300, 'elmo': 256, 'bert': 768}

# set device to GPU if available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
