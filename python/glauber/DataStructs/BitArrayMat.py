import bitarray 
from bitarray import bitarray as ba
import numpy as np

def _fast_index(self, r, c):
    return r * self.ncol + c

def _wraparound_index(self, r, c):

    if r < 0:
        # wrap around over the start (zero)
        r = r + self.nrow
    else:
        # wrap around over the end
        r = r % self.nrow
    
    if c < 0:
        c = c + self.ncol
    else:
        c = c % self.ncol

    return r * self.ncol + c

class BitArrayMat:
    """Wrapper for Bitarray to allow 2D indexing"""

    def __init__(self, nrow, ncol, list=None, wraparound_indices=False) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.size = nrow * ncol
        if list is not None:
            assert nrow * ncol == len(list)
            self.arr = ba(list)
        else:
            self.arr = ba(self.size)

        if wraparound_indices:
            self.idx = _wraparound_index
            self.wraparound_indices = True 
        else:
            self.idx = _fast_index
            self.wraparound_indices = False

    def set_wraparound(self, wrap):
        if wrap:
            self.idx = _wraparound_index
            self.wraparound_indices = True
        else:
            self.idx = _fast_index
            self.wraparound_indices = False
        
    
    def get_with_slice(self, slice):
        return self.arr[slice]

    def __getitem__(self, key):
        # x , y is row, col
        x, y = key
        flat_idx = self.idx(self, x, y)
        return self.arr[flat_idx]

    def __setitem__(self, key, value):
        # x, y is row, col
        x, y = key
        flat_idx = self.idx(self, x, y)
        self.arr[flat_idx] = value

    def count(self, i):
        assert i == 0 or i == 1
        return self.arr.count(i)
    
    def __str__(self) -> str:
        result = ""

        for row in range(self.nrow):
            result += self.arr[row*self.ncol:(row+1)*self.ncol].to01()
            result += "\n"

        return result
    
    def debug_string(self, index):
        # x, y is row, col
        x, y = index
        result = ""

        for row in range(self.nrow):
            row_str = self.arr[row*self.ncol:(row+1)*self.ncol].to01()
            if row == x:
                result += row_str[:y] + "*" + row_str[y+1:]
            else:
                result += row_str
            result += "\n"

        return result
    
    def debug_print(self, index):
        print(self.debug_string(index))
    
    @property
    def shape(self):
        return (self.nrow, self.ncol)
    
    def export_to_file(self, path):
        with open(path, "wb") as f:
            self.arr.tofile(f)

    def load_from_file(self, path):
        self.arr = ba()
        with open(path, "rb") as f:
            self.arr.fromfile(f)
        self.arr = self.arr[:self.nrow * self.ncol].copy()

    def to_numpy(self):
        try:
            bits = np.array(self.arr.tolist())
        except MemoryError as e:
            print(e)
        non_pad = bits[:self.ncol * self.nrow]
        return non_pad.reshape((self.nrow, self.ncol))