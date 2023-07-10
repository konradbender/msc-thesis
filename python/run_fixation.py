import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
import datetime
import os
import glauber2D
import glauberFast


if __name__ == "__main__":

    np.random.seed(0)

    now = datetime.datetime.now()
    time_string = now.strftime('%m%d_%H%M')

    os.makedirs("./fixation-results", exist_ok=True)

    logging.basicConfig(level=logging.ERROR)

    # dimension of the lattice B'
    N_OUTER = 300
    # dimension of the inner lattice B
    N_INTERIOR = 280
    
    # number of time steps for each glauber dynamics iteration
    T = int(2000)

    # number of times we run glauber dynamcis for each probability
    ITERATIONS = 10

    # tolerance for fixation
    TOL = 0.85

    epsilons = [0.01, 0.025, 0.05, 0.075, 0.1]
    probs =  [0.5 + x for x in epsilons] + [0.7, 0.8, 0.9]
    probs.sort()

    results = []

    with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for prob in probs:
            result = executor.submit(glauber2D.run_fixation_simulation, N_OUTER, N_INTERIOR, prob, T, ITERATIONS, TOL)
            futures.append(result)
        
        print("submitted all jobs")
        
        for future in fts.as_completed(futures):
            results.append(future.result())
            prob = future.result()["p"]
            print("results are in for probabiliy: ", prob)

    results.insert(0, {"n_outer": N_OUTER, "n_inner": N_INTERIOR, "t": T, "tol": TOL, "iterations": ITERATIONS,
                       "time_string": time_string})
    
    json.dump(results, open(f"./fixation-results/results-100k-{time_string}.json", "w"))

    print("finished", time_string, ", took ", datetime.datetime.now() - now, " to run")

