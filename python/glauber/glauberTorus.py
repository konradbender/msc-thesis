from typing import Any
from .glauberDynIndices import GlauberSimDynIndices
from .glauberFixIndices import GlauberSimulatorFixIndices
from .glauberSim import GlauberSim
from .DataStructs.ListDict import ListDict
import numpy as np


class GlauberFixedIndexTorus(GlauberSimulatorFixIndices):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["boundary"] = "random"
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.wrap_indices = True

    def setup_indices(self) -> None:
        # list of indices we want to look at -> all except boundary points
        # remember that self.t is the number of iterations
        # Hence may not have index zero or the last elements
        self.indices = np.random.randint(0, self.n_outer , size=2 * self.t).reshape((self.t, 2))
        
    def setup_indices(self) -> None:
        # list of indices we want to look at -> all except boundary points
        # remember that self.t is the number of iterations
        # Hence may not have index zero or the last elements
        self.indices = np.random.randint(0, self.n_outer, size=2 * self.t).reshape((self.t, 2))

    
class GlauberDynIndexTorus(GlauberSimDynIndices):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["boundary"] = "random"
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.wrap_indices = True

    def setup_indices(self) -> None:
        self.indices = ListDict(self.n_outer, self.n_outer)

        for i in range(0, self.n_outer):
            for j in range(0, self.n_outer):
                nb_sum = (
                    self.matrix[i - 1, j]
                    + self.matrix[i + 1, j]
                    + self.matrix[i, j - 1]
                    + self.matrix[i, j + 1]
                )

                # if more than half of neighbors are different, can flip
                if nb_sum > 2 and self.matrix[i, j] == 0:
                    self.indices.add((i, j))
                elif nb_sum < 2 and self.matrix[i, j] == 1:
                    self.indices.add((i, j))

                # if there is a tie, can also flip.
                elif nb_sum == 2:
                    self.indices.add((i, j))   

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
            
            # allow neighbors one unit wrapped around to be added
            if -1 <= neighbor[0] <= self.n_outer  and -1 <= neighbor[1] <= self.n_outer:
            
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


    