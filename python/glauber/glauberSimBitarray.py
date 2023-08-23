from glauber.glauberSim import GlauberSim
import json
from abc import ABC, abstractmethod
from .DataStructs.BitArrayMat import BitArrayMat
from bitarray import bitarray as ba
import numpy as np
import os


DEBUG = False


class GlauberSimBitArray(GlauberSim, ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(f"{self.bitmap_dir}/params.json", "w") as f:
            params = kwargs.copy()
            params.update({"n_outer": self.n_outer})
            json.dump(params, f)

    def setup_matrix(self) -> None:
        self.matrix = np.random.binomial(n=1, p=self.p, size=self.n_outer**2).astype(np.bool_)
        self.matrix = self.matrix.reshape((self.n_outer, self.n_outer))
        

        if self.boundary == 0 or self.boundary == 1:
            self.matrix[0, :] = self.boundary
            self.matrix[-1, :] = self.boundary
            self.matrix[:, 0] = self.boundary
            self.matrix[:, -1] = self.boundary
            self.logger.info(f"Set up boundary as {self.boundary}")
        elif self.boundary == "random":
            self.logger.info("Setting up random boundary")
        else:
            self.error(f"Boundary {self.boundary} not recognized, leaving random")

        self.matrix = BitArrayMat(self.n_outer, self.n_outer, self.matrix.flatten().tolist())

    def setup_interior_mask(self) -> None:
        # make a bitarray for the mask for the inner lattice
        interior_mask = np.zeros((self.n_outer, self.n_outer), dtype=np.bool_)
        
        # set true for n_interior, which is where we want to "sum up" to determine fixation
        interior_mask[self.padding:-self.padding, self.padding:-self.padding] = True

        # make a bitarray
        self.interior_mask =  ba(interior_mask.flatten().tolist())
        if DEBUG:
            self.logger.debug(f"Interior Mask: \n {self.interior_mask.to01()}")

    def load_checkpoint_index(self) -> np.ndarray:
        self.logger.info("Loading checkpoint index")
        last_index = self.checkpoint_file.split("-")[-1].split(".")[0]
        last_index = int(last_index)
        self.logger.info(f"Last index: {last_index}")
        return last_index

    def save_bitmap(self, iter: int) -> None:
        self.logger.debug(f"saving bitmap for iteration {iter}")
        self.matrix.export_to_file(self.bitmap_dir + f"/iter-{iter}.bmp")

    def load_checkpoint_matrix(self) -> BitArrayMat:
        self.logger.info(f"Loading checkpoint matrix from {self.checkpoint_file}")
        matrix = BitArrayMat(self.n_outer, self.n_outer)
        try:
            matrix.load_from_file(self.checkpoint_file)
        except FileNotFoundError:
            self.logger.error("Checkpoint file not found")
            raise FileNotFoundError
        return matrix
    
    def sum_ones(self) -> int:
        return self.matrix.arr[self.interior_mask].count(1)