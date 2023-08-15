import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import matplotlib.pyplot as plt
import json
import datetime
import os
import sys
import argparse

from glauberFixIndices import GlauberSimulatorFixIndices

from copy import deepcopy


def run_traces(n_interior, padding, t, tol, iterations, p, results_dir):

    
    futures = []
    with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

        for i in range(iterations):
            sim = GlauberSimulatorFixIndices(padding = padding,
                                            n_interior = n_interior,
                                            p = p,
                                            t = t,
                                            tol = tol,
                                            results_dir = result_dir + 'rep-' + str(i),
                                            save_bitmaps_every=100_000
                                            )
            future = executor.submit(sim.run_single_glauber, True)
            futures.append(future)

        for future in fts.as_completed(futures):
            pass


if __name__ == "__main__":
    np.random.seed(0)

    now = datetime.datetime.now()
    time_string = now.strftime("%m%d_%H-%M-%S")

    os.makedirs("./results", exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    parser=argparse.ArgumentParser()

    parser.add_argument("--t", help="Number of iterations")
    parser.add_argument("--n", help="Number of repetitions (how many times we run each)")

    args=parser.parse_args()
    
    # number of time steps for each glauber dynamics iteration
    T = int(args.t)

    # number of times we run glauber dynamcis for each probability
    ITERATIONS = int(args.n)

    # dimension of the lattice B
    N_INTERIOR = 500
    # dimension of the padding around B
    PADDING = 10
    
    # tolerance for fixation
    TOL = 0.85

    P = 0.505

    result_dir = "./results/" + time_string + "/"

    run_traces(n_interior=N_INTERIOR, padding=PADDING, t=T, tol=TOL, 
               iterations=ITERATIONS, p=P, results_dir = result_dir)

    params = {
        "n_interior": N_INTERIOR,
        "padding": PADDING,
        "t": T,
        "tol": TOL,
        "iterations": ITERATIONS,
        "time_string": time_string,
        "p": P,
    }  

    with open(result_dir + "iterative-params.json", "w") as f:
        json.dump(params, f)

    logging.info("finished " + time_string + ", took " + str(datetime.datetime.now() - now) + " to run")
    logging.info(f"saved results in directory {result_dir}")
