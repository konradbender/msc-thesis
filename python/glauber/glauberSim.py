from abc import ABC, abstractmethod
import numpy as np
import os
import datetime
import json
import logging
import random
import sys 

import itertools

DEBUG = False
LOGGING_STEP = 100_000


class GlauberSim(ABC):
    def __init__(
        self,
        n_interior: np.int64,
        p: np.float64,
        t: np.int64,
        tol: np.float64,
        boundary = 1,
        padding: np.int64 = None,  # for new code
        n_outer: np.int64 = None,  # kept for depreceated compatibility
        save_bitmaps_every=None,
        results_dir=None,
        random_seed=None,
        checkpoint_file=None
    ) -> None:
        """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
        with probability p of initializing a vertex to 1

        Parameters
        ----------
        n_interior : int
            dimension of inner lattice
        p : float
            probability that vertices are -1 at t=0
        t : int
            number of vertex-updates to perform
        tol : float
            minimum share of -1 vertices to reach to declare fixation
        padding : int
            number of vertices beyond the border of n_interior*n_interior matrix
        save_bitmaps_every : int
            number of iterations after which to save a bitmap of the matrix. If None, is never saved
        results_dir : str
            directory to save results in. If None, results are saved in a directory named after the current time
        random_seed : int
            random seed to use for numpy
        checkpoint_file : str
            path to a checkpoint file to start from. If None, starts from random initialization
        boundary: int or str
            If int, boundary will be set to that value else pass string "random" to set random boundary
        """
        self.results_dir = results_dir

        if self.results_dir is None:
            now = datetime.datetime.now()
            time_string = now.strftime("%m%d_%H-%M-%S")
            self.results_dir = f"./results/{time_string}/"
            logging.info(f"results_dir not specified, using {self.results_dir}")
        else:
            logging.info(f"results_dir specified, using {self.results_dir}")

        os.makedirs(self.results_dir, exist_ok=True)

        self.bitmap_dir = f"{self.results_dir}/bitmap_results/"
            
        os.makedirs(self.bitmap_dir, exist_ok=True)

        # create a new logging file for each instance
        pid = os.getpid()
        self.logger = logging.getLogger(__name__ + '.' +  str(pid))

        if DEBUG:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename=f'{self.results_dir}/log-{pid}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)


        if padding is None and n_outer is None:
            raise ValueError("padding or n_outer must be specified")

        if padding is not None:
            self.padding = padding
            self.n_outer = n_interior + 2 * padding

        if n_outer is not None:
            raise DeprecationWarning("n_outer is deprecated, use padding instead")
            self.n_outer = self.n_outer
            self.padding = (self.n_outer - n_interior) // 2

        self.n_interior = n_interior
        self.p = p
        self.t = t
        self.tol = tol
        self.save_bitmaps_every = save_bitmaps_every
        self.checkpoint_file = checkpoint_file
        self.boundary = boundary

        parameters = {
            "n_interior": self.n_interior,
            "p": self.p,
            "t": self.t,
            "tol": self.tol,
            "padding": self.padding,
            "save_bitmaps_every": self.save_bitmaps_every,
            "results_dir": self.results_dir,
        }
        with open(f"{self.results_dir}/simulation-params.json", "w") as f:
            json.dump(parameters, f)

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            self.logger.warning("random seed set to " + str(random_seed))
        else:
            np.random.seed(os.getpid())
            random.seed(os.getpid())
            self.logger.warning("random seed set to " + str(os.getpid()))

        self.matrix = None
        self.indices = None
        self.interior_mask = None

    @property        
    def checkpoint_available(self):
        if self.checkpoint_file is None:
            return False
        return True
    
    def teardown_sim(self):
        """teardown_sim is called after each simulation run to reset the state of the simulator"""
        self.matrix = None
        self.indice = None
        self.interior_mask = None
    
    @abstractmethod
    def load_checkpoint_index(self) -> np.ndarray:
        """ loads the index from the checkpoint file name - note that it must be in the format '*-<index>.<ending>'"""
        raise NotImplementedError()
    
    @abstractmethod    
    def setup_matrix(self) -> None:
        """setup_matrix initializes the matrix to be used in the simulation. Needs to be indexable by [i, j]"""
        raise NotImplementedError()

    @abstractmethod
    def setup_interior_mask(self) -> None:
        """setup_interior_mask initializes the interior mask to be used in the simulation No requirements, but needs to work with
        self.sum_ones()"""
        raise NotImplementedError()
    
    @abstractmethod
    def sum_ones(self) -> int:
        """sum_ones returns the number of ones in the interior of the matrix"""
        raise NotImplementedError()

    @abstractmethod
    def setup_indices(self) -> None:
        """setup_indices initializes the indices to be used in the simulation. Needs to be indexable by [i] and return a tuple"""
        # TODO make method get_next_index() that returns the next index to be updated
        raise NotImplementedError()
    
    @abstractmethod
    def get_index(self, i) -> tuple:
        """get_index returns the index at position for the ith update"""
        raise NotImplementedError()

    @abstractmethod
    def remove_vertex_from_indices(self, index: tuple) -> None:
        """removes the given index from the indices to be updated"""
        raise NotImplementedError()

    @abstractmethod
    def add_dyn_neighbors_to_indices(self, index: tuple) -> None:
        """adds the dynamic (i.e., not fixated) neighbors of the given index to the indices to be updated"""
        raise NotImplementedError()

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

        self.setup_matrix()

        # this is the padding between inner and outer lattice - 
        # has nothing to do with the boundary condition
        self.logger.debug(f"buffer: {self.padding}")

        self.setup_interior_mask()

        # this is the case if all interior vertices were one, disregarding the 
        # possibility of setting some tolerance
        target = self.n_interior**2
        self.logger.debug(f"Target: {target}")

        vector = np.ones(self.t) * np.int64(-1)

        iterations = 0
        fixation = False

        # Do the warmstarting here

        if self.checkpoint_available:
            self.logger.info("Checkpoint available, trying to load matrix and index")
            try:
                self.matrix = self.load_checkpoint_matrix()
                last_index = self.load_checkpoint_index()
            except FileNotFoundError as e:
                self.logger.error("Checkpoint file not found, starting from scratch")
                last_index = -1
                pass
            
        else:
            last_index = -1

        self.setup_indices()

        self.logger.info("Starting Glauber Simulation at index " + str(last_index + 1))

        # the index is (row, column)
        for i in itertools.islice(range(0, self.t), last_index + 1, None):
            """Updates the vertex at index in the matrix""" 

            if len(self.indices) > 0:
                index = self.get_index(i)
            else:
                self.logger.info("No more indices available, breaking")
                iterations = i
                break

            nb_left = self.matrix[index[0], index[1] - 1]
            nb_right = self.matrix[index[0], index[1] + 1]
            nb_up = self.matrix[index[0] - 1, index[1]]
            nb_down = self.matrix[index[0] + 1, index[1]]              

            nb_sum = (
                nb_left + nb_right + nb_up + nb_down
            )
       
            if nb_sum > 2:
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + self.matrix.debug_string(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"setting vertex at index {index} to 1")

                self.matrix[index[0], index[1]] = 1
                # more than 2 neighbors are 1, so it is fixated
                self.remove_vertex_from_indices(index)
                
            # d is half the number of neighbors
            if nb_sum < 2:
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + self.matrix.debug_string(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"setting vertex at index {index} to 0")

                self.matrix[index[0], index[1]] = 0
                # more than 2 neighbors are 0, so it is fixated
                self.remove_vertex_from_indices(index)
                    

            if nb_sum == 2:
                # flip coin
                z = np.random.binomial(n=1, p=np.float64(0.5))
                if DEBUG:
                    self.logger.debug(f"Matrix before update at index {index}: \n" + self.matrix.debug_string(index))
                    self.logger.debug(f"Sum of neighbors for index {index}: {nb_sum}")
                    self.logger.debug(f"flipping coin for vertex at index {index}, result is {z}")
                
                self.matrix[index[0], index[1]] = z
                # do not add vertex to list of those that can flip because its neighbors are still tied

            summed_array = self.sum_ones()

            vector[i] = summed_array / target

            if self.save_bitmaps_every is not None and (i % self.save_bitmaps_every == 0):
                self.save_bitmap(i)

            if summed_array >= self.tol * target: #  only hit max once
                self.logger.info(f"Fixation at +1 at iteration {i}. Share of 1 is {summed_array / target}.")
                self.logger.debug(f"String Representation of Matrix: \n {str(self.matrix)}")
                fixation = True
                iterations = i
                break
            elif summed_array <= (1 - self.tol) * target: # only hit may once
                self.logger.info(f"Fixation at -1 at iteration {i}. Share of 1 is {summed_array / target}.")
                self.logger.debug(
                              f"String Representation of Matrix: \n {str(self.matrix)}")
                fixation = False
                iterations = i
                break

            if (i % LOGGING_STEP == 0) and verbose:
                self.logger.info(f"iteration: {i} share of 1 is: {summed_array / target}")
                self.logger.info(f"Number of vertices available for update: {len(self.indices)}")

            self.add_dyn_neighbors_to_indices(index)

            if DEBUG:
                self.logger.debug(f"Number of vertices available for update: {len(self.indices)}")

        # end glauber for loop

        if not fixation and len(self.indices) > 0:
            iterations = self.t

        # for good measure, always save last bitmap
        self.save_bitmap(iterations)
        
        result =  {}

        result.update({"fixation": fixation,
                        "iterations":iterations, 
                        "vector": vector.tolist()})
        self.logger.info(f"Result: fixation: {fixation}, iterations: {iterations}")

        with open(f"{self.results_dir}/result-dict.json", "w") as f:
            json.dump(result, f)

        self.teardown_sim()

        return result

        
