import numpy as np
import logging
import json
import datetime
import os
import logging
import itertools
import inspect

from bitarray import bitarray as ba
import bitarray

from .DataStructs.BitArrayMat import BitArrayMat
from .glauberSimBitarray import GlauberSimBitArray


DEBUG = False



class GlauberSimulatorFixIndices(GlauberSimBitArray):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger.info(f"Initializing {type(self).__name__} with parameters: {kwargs}." + 
                     f" Running on PID {os.getpid()}")

    def setup_indices(self) -> None:
        # list of indices we want to look at -> all except boundary points
        # remember that self.t is the number of iterations
        # Hence may not have index zero or the last elements
        self.indices = np.random.randint(1, self.n_outer - 1, size=2 * self.t).reshape((self.t, 2))

    def get_index(self, i) -> tuple:
        return self.indices[i]

    def remove_vertex_from_indices(self, index: tuple) -> None:   
        pass

    def add_dyn_neighbors_to_indices(self, index: tuple) -> None:
        pass

    

    


    
