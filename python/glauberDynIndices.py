import numpy as np
import logging
import json
import datetime
import os
from DataStructs.ListDict import ListDict
from DataStructs.BitArrayMat import BitArrayMat
from glauberSimBitarray import GlauberSimBitArray


from bitarray import bitarray as ba
import bitarray

LOGGING_STEP = 1000


class GlauberSimDynIndices(GlauberSimBitArray):

    def __init__(self, purge_interval = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger.info(f"Initializing GlauberSimDynIndices with parameters: {kwargs}." + 
                     f" Running on PID {os.getpid()}")
        self.purge_interval = purge_interval

    
    def setup_indices(self) -> None:
        self.indices = ListDict(self.n_outer, self.n_outer)

        for i in range(1, self.n_outer - 1):
            for j in range(1, self.n_outer - 1):
                nb_sum = (
                    self.matrix[i - 1, j]
                    + self.matrix[i + 1, j]
                    + self.matrix[i, j - 1]
                    + self.matrix[i, j + 1]
                )

                # if more than half of neighbors are different, can flip
                if nb_sum == 3 and self.matrix[i, j] == 0:
                    self.indices.add((i, j))
                elif nb_sum == 1 and self.matrix[i, j] == 1:
                    self.indices.add((i, j))

                # if there is a tie, can also flip.
                elif nb_sum == 2:
                    self.indices.add((i, j))    

    # implement all abstract methods from GlauberSi   
    
    def get_index(self, i) -> tuple:
        """get_index returns the index at position for the ith update"""
        return self.indices.choose_random_item()

    
    def remove_vertex_from_indices(self, index: tuple) -> None:
        """removes the given index from the indices to be updated"""
        self.indices.remove(index)
 
    def add_dyn_neighbors_to_indices(self, index: tuple) -> None:
        """adds the dynamic (i.e., not fixated) neighbors of the given index to the indices to be updated
        TODO: An alternative idea would be to always add all neighbors, but to periodically remove fixated ones, purge after a certain number has been requested, etc.
        """
        neighbors = (
                        (index[0] + 1, index[1]),
                        (index[0], index[1] + 1),
                        (index[0] - 1, index[1]),
                        (index[0], index[1] - 1),
                        )
        
        for neighbor in neighbors:
            if 0 < neighbor[0] < self.n_outer - 1 and 0 < neighbor[1] < self.n_outer - 1:
                x = neighbor[0]
                y = neighbor[1]
                nb_sum = (
                    self.matrix[x - 1, y]
                    + self.matrix[x + 1, y]
                    + self.matrix[x , y - 1]
                    + self.matrix[x , y + 1]
                )

                # if more than half of neighbors are different, can flip
                if nb_sum > 2 and self.matrix[x, y] == 0:
                    self.indices.add(neighbor)
                elif nb_sum < 2 and self.matrix[x, y] == 1:
                    self.indices.add(neighbor)
                # if there is a tie, can also flip.
                elif nb_sum == 2:
                    self.indices.add(neighbor)  