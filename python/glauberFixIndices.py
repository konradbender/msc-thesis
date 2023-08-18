import numpy as np
import logging
import json
import datetime
import os
import logging
import itertools

from bitarray import bitarray as ba
import bitarray

from DataStructs.BitArrayMat import BitArrayMat
from glauberSim import GlauberSim


LOGGING_STEP = 10000
BOUNDARY = 1
DEBUG = False



class GlauberSimulatorFixIndices(GlauberSim):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.logger.info(f"Initializing GlauberSimulatorFixIndices with parameters: {kwargs}." + 
                     f" Running on PID {os.getpid()}")

        self.bitmap_dir = f"{self.results_dir}/bitmap_results/"
            
        os.makedirs(self.bitmap_dir, exist_ok=True)

        with open(f"{self.bitmap_dir}/params.json", "w") as f:
            params = kwargs.copy()
            params.update({"n_outer": self.n_outer})
            json.dump(params, f)


    def save_bitmap(self, matrix: BitArrayMat, iter: int) -> None:
        self.logger.debug(f"saving bitmap for iteration {iter}")
        matrix.export_to_file(self.bitmap_dir + f"/iter-{iter}.bmp")

    def load_checkpoint_matrix(self) -> BitArrayMat:
        self.logger.info("Loading checkpoint matrix")
        matrix = BitArrayMat(self.n_outer, self.n_outer)
        matrix.load_from_file(self.checkpoint_file)
        return matrix
    
    def load_checkpoint_index(self) -> np.ndarray:
        self.logger.info("Loading checkpoint index")
        last_index = self.checkpoint_file.split("-")[-1].split(".")[0]
        last_index = int(last_index)
        self.logger.info(f"Last index: {last_index}")
        return last_index
        

    def run_single_glauber(
        self,
        verbose: bool = False,
        *args, **kwargs
    ) -> dict:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1
        
        Parameters
        ----------
        verbose : bool
            if print statements should be performed
        """
        
        self.logger.info(f"Simulation Running on proccess with PID {os.getpid()}")

        matrix = np.random.binomial(n=1, p=self.p, size=self.n_outer**2).astype(np.bool_)
        matrix = matrix.reshape((self.n_outer, self.n_outer))
        matrix[0, :] = BOUNDARY
        matrix[-1, :] = BOUNDARY
        matrix[:, 0] = BOUNDARY
        matrix[:, -1] = BOUNDARY

        matrix = BitArrayMat(self.n_outer, self.n_outer, matrix.flatten().tolist())

        # this is the padding between inner and outer lattice - 
        # has nothing to do with the boundary condition
        self.logger.debug(f"buffer: {self.padding}")

        # make a bitarray for the mask for the inner lattice
        interior_mask = np.zeros((self.n_outer, self.n_outer), dtype=np.bool_)
        
        # set true for n_interior, which is where we want to "sum up" to determine fixation
        interior_mask[self.padding:-self.padding, self.padding:-self.padding] = True

        # make a bitarray
        interior_mask =  ba(interior_mask.flatten().tolist())
        if DEBUG:
            self.logger.debug(f"Interior Mask: \n {interior_mask.to01()}")

        # this is the case if all interior vertices were one, disregarding the 
        # possibility of setting some tolerance
        target = self.n_interior**2
        self.logger.debug(f"Target: {target}")

        vector = np.ones(self.t) * np.int64(-1)

        iterations = 0
        fixation = False

        # Do the warmstarting here

        if self.checkpoint_available():
            self.logger.info("Checkpoint available, loading matrix and index")
            matrix = self.load_checkpoint_matrix()
            last_index = self.load_checkpoint_index()
        else:
            last_index = -1

        
        # list of indices we want to look at -> all except boundary points
        # remember that self.t is the number of iterations
        # Hence may not have index zero or the last elements
        indices = np.random.randint(1, self.n_outer - 1, size=2 * self.t).reshape((self.t, 2))
        self.logger.info("Starting Glauber Simulation at index " + str(last_index + 1))

        # the index is (row, column)
        for i, index in itertools.islice(enumerate(indices), last_index + 1, None):
            """Updates the vertex at index in the matrix"""               

            nb_sum = (
                matrix[index[0] + 1, index[1]] +
                matrix[index[0], index[1] + 1] +
                matrix[index[0] - 1, index[1]] +
                matrix[index[0], index[1] - 1] 
            )
                

            # d is half the number of neighbors
            if nb_sum > 2:
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"setting vertex at index {index} to 1")

                matrix[index[0], index[1]] = 1
                
            # d is half the number of neighbors
            if nb_sum < 2:
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"setting vertex at index {index} to 0")

                matrix[index[0], index[1]] = 0
                    

            if nb_sum == 2:
                # flip coin
                z = np.random.binomial(n=1, p=np.float64(0.5))
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"flipping coin for vertex at index {index}, result is {z}")

                matrix[index[0], index[1]] = z

            summed_array = matrix.arr[interior_mask].count(1)

            vector[i] = summed_array / target

            if self.save_bitmaps_every is not None and (i % self.save_bitmaps_every == 0):
                self.save_bitmap(matrix, i)

            if summed_array >= self.tol * target: #  only hit max once
                self.logger.info(f"Fixation at +1 at iteration {i}. Share of 1 is {summed_array / target}.")
                self.logger.debug(f"String Representation of Matrix: \n {str(matrix)}")
                fixation = True
                iterations = i
                break
            elif summed_array <= (1 - self.tol) * target: # only hit may once
                self.logger.info(f"Fixation at -1 at iteration {i}. Share of 1 is {summed_array / target}." +
                              f"String Representation of Matrix: \n {str(matrix)}")
                fixation = False
                iterations = i
                break

            if (i % LOGGING_STEP == 0) and verbose:
                self.logger.info(f"iteration: {i} share of 1 is: {summed_array / target}")

        # end glauber for loop

        if not fixation:
            iterations = self.t

        # for good measure, always save last bitmap
        self.save_bitmap(matrix, iterations)
        
        result =  {}

        result.update({"fixation": fixation,
                        "iterations":iterations, 
                        "vector": vector.tolist()})
        self.logger.info(f"Result: fixation: {fixation}, iterations: {iterations}")

        with open(f"{self.results_dir}/result-dict.json", "w") as f:
            json.dump(result, f)

        return result


    
