import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import json
import datetime
import os
import sys
import argparse

from glauberFixIndices import GlauberSimulatorFixIndices

RESULT_DIR = "./results/"
LOG_DIR = "./logs/"

def check_for_checkpoints(results_dir, params_dict):
    """If the last run in the results dir had the same parameters, will return that directory name and else None"""

    format = "%m%d_%H-%M-%S"

    contents = os.listdir(results_dir)
    runs = []
    for c in contents:
        try:
            dt = datetime.datetime.strptime(c, format)  
        except ValueError as e:
            logging.info("found non-date directory: " + c)
            break
        files_of_run = os.listdir(results_dir + c)
        if "iterative-params.json" not in files_of_run:
            logging.info("found directory without iterative-params.json: " + c)
        else:
            runs.append(dt)
    
    if len(runs) == 0:
        logging.info("no runs found in directory " + results_dir)
        return None 
    
    runs.sort()
    last_run = runs[-1]
    last_run_str = datetime.datetime.strftime(last_run, format)
    logging.info("last run in directory " + results_dir + " was " + str(last_run))

    last_params = {}
    with open(results_dir + datetime.datetime.strftime(last_run, format) + "/iterative-params.json") as f:
        last_params = json.load(f)
    for key, value in params_dict.items():
        if value != last_params[key]:
            logging.info("last run had different parameters")
            return None
    
    logging.info("last run had same parameters, returning checkpoint directories")

    reps = os.listdir(results_dir + last_run_str + "/")
    reps = [last_run_str + '/' + x for x in reps if x.startswith("rep-")]

    return reps

def create_sim_and_submit(*args, **kwargs):
        sim = GlauberSimulatorFixIndices(*args, **kwargs)
        result = sim.run_single_glauber(verbose=False)
        return result

def run_traces(n_interior, padding, t, tol, iterations, p, results_dir, checkpoint_int, warmstarts):
    
    futures = []
    with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

        for i in range(iterations):
            future = executor.submit(create_sim_and_submit,
                                    padding = padding,
                                    n_interior = n_interior,
                                    p = p,
                                    t = t,
                                    tol = tol,
                                    results_dir = result_dir + 'rep-' + str(i),
                                    save_bitmaps_every =checkpoint_int,
                                    random_seed = i,
                                    checkpoint_file = warmstarts[i]
                                    )
            
            futures.append(future)

        for future in fts.as_completed(futures):
            result = future.result()


if __name__ == "__main__":
    np.random.seed(0)

    now = datetime.datetime.now()
    time_string = now.strftime("%m%d_%H-%M-%S")

    logging.basicConfig(level=logging.INFO)

    logging.info("starting " + time_string)
    logging.info("number of cores: " + str(mp.cpu_count()))

    parser=argparse.ArgumentParser()

    parser.add_argument("--t", help="Number of iterations")
    parser.add_argument("--n", help="Number of repetitions (how many times we run each)")
    parser.add_argument("--checkpoint", help="Number of steps between checkpoint saves")
    parser.add_argument("--n_interior", help="Size of the interior of the lattice")
    parser.add_argument("--padding", help="Size of the padding around the lattice")
    parser.add_argument("--force_new", help="Can surpress checkpoint loading",
                        action="store_true")

    args=parser.parse_args()
    logging.info("Started run for multiple traces with args" + str(args))
    
    # number of time steps for each glauber dynamics iteration
    T = int(args.t)

    # number of times we run glauber dynamcis for each probability
    ITERATIONS = int(args.n)

    # dimension of the lattice B
    N_INTERIOR = int(args.n_interior)
    # dimension of the padding around B
    PADDING = int(args.padding)

    CHECKPOINT_INTERVAL = int(args.checkpoint)

    FORCE_NEW = bool(args.force_new)
    
    # tolerance for fixation
    TOL = 0.85

    P = 0.505

    params = {
        "n_interior": N_INTERIOR,
        "padding": PADDING,
        "t": T,
        "tol": TOL,
        "iterations": ITERATIONS,
        "p": P,
        "iterations": ITERATIONS,
    }  

    result_dir = RESULT_DIR  + time_string + "/"
    os.makedirs(result_dir, exist_ok=True)

    if not FORCE_NEW:
        checkpoint_runs = check_for_checkpoints(RESULT_DIR , params)
    else:
        checkpoint_runs = None
        logging.info("Force new was set, so not loading checkpoints.")

    warmstarts = [None] * ITERATIONS

    if checkpoint_runs is not None and len(checkpoint_runs) > 0:
        logging.info("found checkpoint directories " + str(checkpoint_runs))
        params["warmstart_from"] = checkpoint_runs
        if len(checkpoint_runs) < ITERATIONS:
            logging.info("checkpoint does not have same number of runs, starting all from the same")
            warmstarts = [checkpoint_runs[0] + 'bitmap_results/iter-' + str(T) + '.bmp'] * ITERATIONS
        elif len(checkpoint_runs) == ITERATIONS:
            logging.info("checkpoint has same number of runs, starting each from its own")
            warmstarts = [RESULT_DIR + x + '/bitmap_results/iter-' + str(T) + '.bmp' for x in checkpoint_runs]
    
    
    run_traces(n_interior=N_INTERIOR, padding=PADDING, t=T, tol=TOL, 
            iterations=ITERATIONS, p=P, results_dir = result_dir, 
            checkpoint_int=CHECKPOINT_INTERVAL, warmstarts = warmstarts)

    params.update({"time_string": time_string})

    with open(result_dir + "iterative-params.json", "w") as f:
        json.dump(params, f)

    logging.info("finished " + time_string + ", took " + str(datetime.datetime.now() - now) + " to run")
    logging.info(f"saved results in directory {result_dir}")
