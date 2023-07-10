import numpy as np
np.random.seed(0)
import logging

def z_d(d, n, p):
    """Returns a d-dimensional tensor of size n, vertices initialized 
    to 1 or 0 with probability p"""
    # gives it out with a buffer, so a 2x2x2 cube would be 3x3x3

    # TODO Boundary conditions? Should we set them to all 1?
    shape = [n+2]*d
    return np.random.binomial(n=1, p=p, size=shape)

def _get_neighbor_indices(d, index):
    """Returns the neighbors of a given index in the lattice"""
    if len(index) != d:
        raise ValueError("index must be same length as lattice shape")
    
    neighbors = []
    
    for i in range(d): # TODO make this faster
        new_index_down = index.copy()
        new_index_down[i] -= 1
        new_index_up = index.copy()
        new_index_up[i] += 1

        neighbors.append(new_index_down)
        neighbors.append(new_index_up)
    
    logging.debug(f"neighbors of {index} are {neighbors}")
    return np.array(neighbors)

def get_random_indices(d, n, t):
    """Returns t random indices for a d-dimensional lattice of size n"""
    return np.random.randint(0, n, size=d*t).reshape((t, d))

def update_vertex(d, lattice, index):
    """Updates the vertex at index in the matrix"""

    neighbors = _get_neighbor_indices(d, index)
    
    # need to add 1 to index because of buffer
    neighbor_indices = neighbors + 1
    
    neighbor_indices = tuple(neighbor_indices.T.reshape((D, 2*D)))
    sum = np.sum(lattice[neighbor_indices])
    logging.debug(f"neighbors of {index} are {lattice[neighbor_indices]}, sum is {sum}")
    # d is half the number of neighbors
    if sum > d:
        # again add one to index because of buffer
        lattice[tuple(index+1)] = 1
        logging.debug(f"updated {index} to 1")
        return
    if sum == d:
        # flip coin
        z = np.random.binomial(n=1, p=0.5)
        lattice[tuple(index+1)] = z
        logging.debug(f"updated {index} to random number {z}")
        return
    else:
        logging.debug(f"did not update {index}")
        return
    

def run_single_glauber(d,n,p,t, vector_path = None):
    """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
    with probability p of initializing a vertex to 1"""
    matrix = z_d(d,n,p)
    indices = get_random_indices(d,n,t)
    interior_indices = tuple([slice(1,-1)]*D)
    if vector_path:
        vector = np.ndarray(shape=t)

    for i, index in enumerate(indices):
        update_vertex(d, matrix, index)
        summed_array = matri
        if sum(matrix[interior_indices].flatten()) == N**D:
            return True
    
    return False

def run_simulation(d,n,p,t, iter):

    fixations = 0
    for i in range(iter):
        fixation = run_single_glauber(d,n,p,t)
        if fixation:
            fixations += 1
    
    return fixations / iter



if __name__ == "__main__":

    D = 3
    N = 10
    T = 10000
    iter = 10

    probs = np.arange(0.4, 0.6, 0.05)

    fixations = {p: run_simulation(D,N,p,T, iter) for p in probs}

    print("Fixation results:")
    for key, value in fixations.items():
        print(f"p={key:.5f}, fixation={value:.5f}")