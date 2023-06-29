import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
import datetime
import os

def squary_boundary_fix(n, p, boundary=1):
    """Returns a 2-dimensional matrix of size n, vertices initialized 
    to 1 or 0 with probability p"""
    # gives it out with a buffer, so a 2x2x2 cube would be 3x3x3

    shape = [n]*2
    matrix =  np.random.binomial(n=1, p=p, size=shape)
    matrix[0,:] = boundary
    matrix[-1,:] = boundary

    return matrix

def squary_boundary_random(n, p, boundary=1):
    """Returns a 2-dimensional matrix of size n, vertices initialized 
    to 1 or 0 with probability p"""
    # gives it out with a buffer, so a 2x2x2 cube would be 3x3x3

    shape = [n]*2
    matrix =  np.random.binomial(n=1, p=p, size=shape)

    return matrix

def get_random_indices(n, t):
    """Returns t random indices for a 2-dimensional lattice of size n, but will not choose 
    one from the boundary"""

    # for randint, the first argument is inclusive, the second is exclusive
    return np.random.randint(1, n-1, size=2*t).reshape((t, 2))
    

def run_single_glauber(n_outer, n_interior, p, t, tol, where_to_fixate='upper'):
    """Runs a simulation of the Glauber dynamics on a d-dimensional lattice of size n
    with probability p of initializing a vertex to 1"""

    t = int(t)

    matrix = squary_boundary_fix(n_outer,p)
    indices = get_random_indices(n_outer,t)
    
    buffer = (n_outer - n_interior) // 2
    interior_indices = tuple([slice(buffer,
                                    -buffer)]*2)

    iterations = 0
    fixation = False
    for index in indices:
        iterations += 1
        """Updates the vertex at index in the matrix"""

        neighbor_indices = np.array([index + np.array([0,1]),
                            index + np.array([1,0]),
                            index + np.array([-1,0]),
                            index + np.array([0,-1]),
                            ])
        
        neighbor_indices = tuple(neighbor_indices.T.reshape((2, 4)))



        nb_sum = np.sum(matrix[neighbor_indices])
        
        # d is half the number of neighbors
        if nb_sum > 2:
            # again add one to index because of buffer
            matrix[tuple(index)] = 1
            
        # d is half the number of neighbors
        if nb_sum < 2:
            # again add one to index because of buffer
            matrix[tuple(index)] = 0

        if nb_sum == 2:
            # flip coin
            z = np.random.binomial(n=1, p=0.5)
            matrix[tuple(index)] = z
            
            
        if where_to_fixate == 'upper' and sum(matrix[interior_indices].flatten()) >= \
            tol*matrix[interior_indices].shape[0]*matrix[interior_indices].shape[1]:
            fixation = True
            break
        elif where_to_fixate == 'lower' and sum(matrix[interior_indices].flatten()) <= \
            (1-tol) * matrix[interior_indices].shape[0]*matrix[interior_indices].shape[1]:
            fixation = True
            break

        if iterations % 1000 == 0:
            print(f"iteration {iterations} of {t} for probability {p}")
    
    result = {"fixation": fixation, "iterations": iterations,
            "p": p, "n_outer": n_outer, "n_interior": n_interior, "t": t, "tol": tol, 
            "where_to_fixate": where_to_fixate}
    print("finished a single run of glauber dynamics with result: ", result,)
    
    return result

def run_simulation(n_outer, n_inner, p, t, iter, tol):

    fixations = 0
    iterations = 0

    for i in range(iter):
        result = run_single_glauber(n_outer, n_inner, p, t, tol)

        if result["fixation"]:
            fixations += 1
            iterations += result["iterations"]


    return {'fixation_rate': fixations / iter,
            'mean_iterations': iterations / iter,
            "p": p, "n_outer": n_outer,"n_inner": n_inner, "t": t}


if __name__ == "__main__":

    np.random.seed(0)

    now = datetime.datetime.now()
    time_string = now.strftime('%m%y_%H%M')

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    logging.basicConfig(level=logging.ERROR)

    # dimension of the lattice B'
    N_OUTER = 300
    # dimension of the inner lattice B
    N_INTERIOR = 280
    
    # number of time steps for each glauber dynamics iteration
    T = int(10)

    # number of times we run glauber dynamcis for each probability
    ITER = 3

    # tolerance for fixation
    TOL = 0.8

    epsilons = [0.01, 0.025, 0.05, 0.075, 0.1]
    probs =  [0.5 - x for x in epsilons] + [0.5 + x for x in epsilons] + [0.7, 0.8, 0.9]
    probs.sort()

    results = []

    with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for prob in probs:
            result = executor.submit(run_simulation, N_OUTER, N_INTERIOR, prob, T, ITER, TOL)
            futures.append(result)
        
        print("submitted all jobs")
        
        for future in fts.as_completed(futures):
            results.append(future.result())
            prob = future.result()["p"]
            print("results are in for probabiliy: ", prob)

    results.insert(0, {"n_outer": N_OUTER, "n_inner": N_INTERIOR, "t": T, "tol": TOL, "iterations": ITER,
                       "time_string": time_string})
    
    json.dump(results, open(f"results/results-100k-{time_string}.json", "w"))


    print("finished, took ", datetime.datetime.now() - now, " to run")



