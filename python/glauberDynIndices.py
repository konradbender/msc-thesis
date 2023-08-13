import numpy as np
import logging
import json
import datetime
import os
from DataStructs.IndexList import IndexList
from DataStructs.BitArrayMat import BitArrayMat
from glauberSim import GlauberSim


from bitarray import bitarray as ba
import bitarray

# TODO make torus
# TODO have the matrix of present vertices in the ListDict -> can be a bitarray

LOGGING_STEP = 1000
BOUNDARY = 1


class GlauberSimDynIndices(GlauberSim):

    def _get_dynamic_indices(self, bitarray_mat: BitArrayMat) -> IndexList:
        result = IndexList(self.n_outer, self.n_outer)

        for i in range(1, self.n_outer - 1):
            for j in range(1, self.n_outer - 1):
                sum = (
                    bitarray_mat[i - 1, j]
                    + bitarray_mat[i + 1, j]
                    + bitarray_mat[i, j - 1]
                    + bitarray_mat[i, j + 1]
                )

                # if more than half of neighbors are different, can flip
                if sum == 3 and bitarray_mat[i, j] == 0:
                    result.add((i, j))
                elif sum == 1 and bitarray_mat[i, j] == 1:
                    result.add((i, j))

                # if there is a tie, can also flip.
                elif sum == 2:
                    result.add((i, j))

        return result
    
    
    def add_neighbors(self, index, list):
        """adds neighbors to the list if they are dynamic"""

        neighbors = (
                            (index[0] + 1, index[1]),
                            (index[0], index[1] + 1),
                            (index[0] - 1, index[1]),
                            (index[0], index[1] - 1),
                        )
        
        for neighbor in neighbors:
            if 0 < neighbor[0] < self.n_outer - 1 and 0 < neighbor[1] < self.n_outer - 1:
                list.add(neighbor)



    def run_single_glauber(self, run_id: int = None, verbose: bool = False) -> dict:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1

        Parameters
        ----------
        run_id : string
            identification of this glauber run to retrieve files saved to disk
        verbose : bool
            if print statements should be performed
        """

        matrix = np.random.binomial(n=1, p=self.p, size=self.n_outer**2).astype(
            np.bool_
        )
        matrix = matrix.reshape((self.n_outer, self.n_outer))
        matrix[0, :] = BOUNDARY
        matrix[-1, :] = BOUNDARY

        matrix = BitArrayMat(self.n_outer, self.n_outer, matrix.flatten().tolist())

        # list of indices we want to look at
        possible_indices = self._get_dynamic_indices(matrix)

        # this is the padding between inner and outer lattice
        buffer = (self.n_outer - self.n_interior) // 2

        # make a bitarray for the mask for the inner lattice
        interior_mask = np.zeros((self.n_outer, self.n_outer), dtype=np.bool_)
        interior_mask[buffer:-buffer, buffer:-buffer] = 1
        interior_mask = ba(interior_mask.flatten().tolist())

        target = self.n_interior**2

        vector = np.ones(self.t) * np.int64(-1)

        iterations = 0
        fixation = np.bool_(False)

        for i in range(self.t):
            """Updates the vertex at index in the matrix"""

            index = possible_indices.choose_random_item()
            possible_indices.remove(index)

            nb_sum = (
                matrix[index[0] + 1, index[1]]
                + matrix[index[0], index[1] + 1]
                + matrix[index[0] - 1, index[1]]
                + matrix[index[0], index[1] - 1]
            )

            # d is half the number of neighbors
            if nb_sum > 2:
                # again add one to index because of buffer
                # if this is true then have to flip
                if matrix[index[0], index[1]] == 0:
                    matrix[index[0], index[1]] = 1
                    self.add_neighbors(index, possible_indices)

            # d is half the number of neighbors
            if nb_sum < 2:
                # again add one to index because of buffer
                # if this is true then have to flip
                if matrix[index[0], index[1]] == 1:
                    matrix[index[0], index[1]] = 0
                    self.add_neighbors(index, possible_indices)

            if nb_sum == 2:
                # flip coin
                z = np.random.binomial(n=1, p=np.float64(0.5))
                matrix[index[0], index[1]] = z
                self.add_neighbors(index, possible_indices)
                possible_indices.add(index)

            summed_array = matrix.arr[interior_mask].count(1)

            vector[i] = summed_array / target

            if summed_array >= self.tol * target:
                fixation = np.bool_(True)
                iterations = i
                break
            elif summed_array <= (1 - self.tol) * target:
                fixation = np.bool_(False)
                iterations = i
                break

            if (i % LOGGING_STEP == 0) and verbose:
                print("iteration:", i, "share of 1 is:", (summed_array / target))

        # end glauber for loop

        result = {}

        result.update(
            {
                "fixation": np.asarray([np.float64(fixation)]),
                "iterations": np.asarray([np.float64(iterations)]),
                "vector": vector,
            }
        )

        return result

    