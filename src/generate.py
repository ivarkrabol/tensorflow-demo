import numpy as np

from config import out


SIZE = 100000

# First bit is overflow
BITS = 3


def to_bit_lists(a):
    return [list(map(float, '{0:064b}'.format(n)[-BITS:])) for n in a]


np.random.seed(1)

left = np.random.randint(0, (1 << (BITS - 1)) - 1, SIZE)
right = np.random.randint(0, (1 << (BITS - 1)) - 1, SIZE)

plus = left + right

x = np.moveaxis(np.array([
    to_bit_lists(left),
    to_bit_lists(right),
]), 0, 1)
np.save(out('data.npy'), x)

y = np.moveaxis(np.array([
    to_bit_lists(plus),
]), 0, 1)
np.save(out('labels.npy'), y)
