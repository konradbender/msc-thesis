from glauberFixIndices import GlauberSimulatorFixIndices
from glauberDynIndices import GlauberSimDynIndices
import logging
logging.basicConfig(level=logging.DEBUG)
import json
import os
import concurrent.futures as fts
import multiprocessing as mp
import socket
import sys
import gitinfo  
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--stem", help="root Dirs of runs to continue")

def continue_run(stem, rep, iter):

    f_old_params = os.path.join(stem, f"rep-{rep}", "simulation-params.json")
    f_old_bitmap = os.path.join(stem, f"rep-{rep}", "bitmap_results", f"iter-{iter}.bmp")
    f_results = os.path.join(stem, f"rep-{rep}", "result-dict.json")

    old_params = json.load(open(f_old_params, "r"))
    old_results = json.load(open(f_results, "r"))

    n_interior = old_params["n_interior"]
    padding = old_params["padding"]
    p = old_params["p"]
    tol = old_params["tol"]
    checkpoint_freq = 1e5

    old_steps = old_results["iterations"]
    more_steps_to_do = 10e6

    
    result_dir = str(os.path.join(stem + '-continuation', f"rep-{rep}"))

    sim = GlauberSimDynIndices(
        n_interior=int(n_interior),
        padding=int(padding),
        p=p,
        t=int(old_steps + more_steps_to_do),
        tol=1,
        results_dir=result_dir,
        checkpoint_file=f_old_bitmap,
        random_seed=i,
        save_bitmaps_every=checkpoint_freq,
    )

    sim.run_single_glauber(verbose=True)


if __name__ == "__main__":


    args=parser.parse_args()

    iters = range(0, 6)
    reps = int(20e6)

    stem = args.stem

    futures = []

    result_dir = stem + '-continuation'
    os.makedirs(result_dir, exist_ok=True)

    # create a new logging file for each instance
    pid = os.getpid()
    logger = logging.getLogger(__name__ + '.' +  str(pid))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename=f'{result_dir}/log-main.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    

    logger.info("running on " + socket.gethostname())

    if gitinfo.get_git_info() is not None:
        logger.info("git info: " + str(gitinfo.get_git_info())) 

    with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as ex:

        for i in iters:
            future = ex.submit(continue_run, stem, i, reps)
            futures.append(future)
            logger.info(f"submitted {i}")

        for future in fts.as_completed(futures):
            result = future.result()

    logger.info("done, saved results in " + result_dir + "")
