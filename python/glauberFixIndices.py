import numpy as np
import logging
import json
import datetime
import os
import logging

from bitarray import bitarray as ba
import bitarray

from DataStructs.BitArrayMat import BitArrayMat
from glauberSim import GlauberSim


LOGGING_STEP = 1000
BOUNDARY = 1
DEBUG = False



class GlauberSimulatorFixIndices(GlauberSim):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logging.info(f"Initializing GlauberSimulatorFixIndices with parameters: {kwargs}")
        if self.save_bitmaps_every is not None:
            now = datetime.datetime.now()
            time_string = now.strftime('%m%d_%H-%M-%S')
            self.bitmap_dir = f"bitmap_results/{time_string}"
            os.makedirs(self.bitmap_dir, exist_ok=True)

            with open(f"{self.bitmap_dir}/params.json", "w") as f:
                params = kwargs.copy()
                params.update({"time_string": time_string})
                params.update({"n_outer": self.n_outer})
                json.dump(params, f)


    def save_bitmap(self, matrix: BitArrayMat, iter: int) -> None:
        logging.debug(f"saving bitmap for iteration {iter}")
        matrix.export_to_file(self.bitmap_dir + f"/iter-{iter}.bmp")
        

    def run_single_glauber(
        self,
        verbose: bool = False,
        *args, **kwargs
    ) -> dict:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1
        
        Parameters
        ----------
        n_outer : int
            dimension of outer lattice
        n_inner : int
            dimension of inner lattice
        p : float
            probability that vertices are -1 at t=0
        t : int
            number of vertex-updates to perform
        tol : float
            minimum share of -1 vertices to reach to declare fixation
        verbose : bool
            if print statements should be performed
        """
        

        matrix = np.random.binomial(n=1, p=self.p, size=self.n_outer**2).astype(np.bool_)
        matrix = matrix.reshape((self.n_outer, self.n_outer))
        matrix[0, :] = BOUNDARY
        matrix[-1, :] = BOUNDARY
        matrix[:, 0] = BOUNDARY
        matrix[:, -1] = BOUNDARY

        matrix = BitArrayMat(self.n_outer, self.n_outer, matrix.flatten().tolist())

        # this is the padding between inner and outer lattice - 
        # has nothing to do with the boundary condition
        logging.debug(f"buffer: {self.padding}")

        # make a bitarray for the mask for the inner lattice
        interior_mask = np.zeros((self.n_outer, self.n_outer), dtype=np.bool_)
        
        # set true for n_interior, which is where we want to "sum up" to determine fixation
        interior_mask[self.padding:-self.padding, self.padding:-self.padding] = True

        # make a bitarray
        interior_mask =  ba(interior_mask.flatten().tolist())
        if DEBUG:
            logging.debug(f"Interior Mask: \n {interior_mask.to01()}")

        # this is the case if all interior vertices were one, disregarding the 
        # possibility of setting some tolerance
        target = self.n_interior**2
        logging.debug(f"Target: {target}")

        vector = np.ones(self.t) * np.int64(-1)

        iterations = 0
        fixation = np.bool_(False)

        # list of indices we want to look at -> all except boundary points
        # remember that self.t is the number of iterations
        # Hence may not have index zero or the last elements
        indices = np.random.randint(1, self.n_outer - 1, size=2 * self.t).reshape((self.t, 2))

        # the index is (row, column)
        for i, index in enumerate(indices):
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
                    logging.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    logging.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    logging.debug(f"setting vertex at index {index} to 1")

                matrix[index[0], index[1]] = 1
                
            # d is half the number of neighbors
            if nb_sum < 2:
                if DEBUG:
                    logging.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    logging.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    logging.debug(f"setting vertex at index {index} to 0")

                matrix[index[0], index[1]] = 0
                    

            if nb_sum == 2:
                # flip coin
                if DEBUG:
                    logging.debug(f"Matrix before update at index {index}: \n" + matrix.debug_print(index))
                    logging.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    logging.debug(f"flipping coin for vertex at index {index}, result is {z}")

                z = np.random.binomial(n=1, p=np.float64(0.5))
                matrix[index[0], index[1]] = z

            summed_array = matrix.arr[interior_mask].count(1)

            vector[i] = summed_array / target

            if summed_array >= self.tol * target: #  only hit max once
                logging.info(f"Fixation at +1 at iteration {i}. Share of 1 is {summed_array / target}.")
                logging.debug(f"String Representation of Matrix: \n {str(matrix)}")
                fixation = np.bool_(True)
                iterations = i
                break
            elif summed_array <= (1 - self.tol) * target: # only hit may once
                logging.info(f"Fixation at -1 at iteration {i}. Share of 1 is {summed_array / target}." +
                              f"String Representation of Matrix: \n {str(matrix)}")
                fixation = np.bool_(False)
                iterations = i
                break

            if (i % LOGGING_STEP == 0) and verbose:
                logging.info(f"iteration: {i} share of 1 is: {summed_array / target}")

            if self.save_bitmaps_every is not None and (i % self.save_bitmaps_every == 0):
                self.save_bitmap(matrix, i)

        # end glauber for loop

        if not fixation:
            iterations = self.t
        
        result =  {}

        result.update({"fixation": fixation,
                        "iterations":iterations, 
                        "vector": vector})
        logging.info(f"Result: {result}")


        return result


    
