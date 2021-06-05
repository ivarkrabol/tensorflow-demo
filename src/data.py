import numpy as np

from config import out


def load_data():
    x = np.load(out('data.npy'))
    y = np.load(out('labels.npy'))

    return x, y
