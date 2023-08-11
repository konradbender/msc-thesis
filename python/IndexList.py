import random
from BitArrayMat import BitArrayMat
from bitarray import bitarray as ba
import numpy as np


class IndexList(object):
    def __init__(self, n_row, n_col):
        self.included = BitArrayMat(n_row, n_col)
        self.included.arr.setall(False)

    def add(self, item):
        self.included[item] = True

    def extend(self, items):
        for item in items:
            self.add(item)

    def remove(self, item):
        self.included[item] = False

    def choose_random_item(self):
        present = self.included.count(True)
        index = random.randint(0, present-1)
        ones = self.included.arr.itersearch(ba('1'))
        ones = list(ones)
        one_d_index = ones[index]
        return (one_d_index // self.included.ncol, one_d_index % self.included.ncol)

if __name__ == '__main__':
    indices = ListDict(10,10)

    indices.add((0, 0))
    indices.add((0, 1))
    indices.add((0, 2))
    indices.add((9, 9))
    indices.add((9, 8))
    indices.add((5, 5))
    print('after adding all')
    print(indices.included)

    indices.remove((0, 0))
    print('after removing (0,0)')
    print(indices.included)

    print(indices.choose_random_item())
    print(indices.choose_random_item())
    print(indices.choose_random_item())
    print(indices.choose_random_item())