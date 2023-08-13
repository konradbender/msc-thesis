import bitarray 
from bitarray import bitarray as ba

class BitArrayMat:
    """Wrapper for Bitarray to allow 2D indexing"""

    def __init__(self, nrow, ncol, list=None) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.size = nrow * ncol
        if list is not None:
            assert nrow * ncol == len(list)
            self.arr = ba(list)
        else:
            self.arr = ba(self.size)
        

    def idx(self, r, c):
        assert 0 <= r < self.nrow
        assert 0 <= c < self.ncol
        return r * self.ncol + c
    
    def get_with_slice(self, slice):
        return self.arr[slice]

    def __getitem__(self, key):
        # x , y is row, col
        x, y = key
        flat_idx = self.idx(x, y)
        # TODO make try-catch and if index out of bouds,
        # return a 1 if index one beyond
        return self.arr[flat_idx]

    def __setitem__(self, key, value):
        # x, y is row, col
        x, y = key
        flat_idx = self.idx(x, y)
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
    
    def debug_print(self, index):
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
    
    @property
    def shape(self):
        return (self.nrow, self.ncol)
    