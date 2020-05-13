

import os

if os.getcwd().split(os.sep)[1] == 'wrk': # if working on Ukko2

    TESTING = True
else:
    TESTING = False

# make sure you are in the project dir (parent of source)
DATA_PATH = os.path.join(os.getcwd(), 'corpus')