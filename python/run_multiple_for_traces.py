import numpy as np
import logging
import concurrent.futures as fts
import multiprocessing as mp
import json
import datetime
import os
import sys
import argparse
import socket
import gitinfo

from glauberFixIndices import GlauberSimulatorFixIndices

RESULT_DIR = "./results/"
LOG_DIR = "./logs/"

parser=argparse.ArgumentParser()

parser.add_argument("--t", help="Number of iterations")
parser.add_argument("--p", help="Probability of +1 at initialization")
parser.add_argument("--n", help="Number of repetitions (how many times we run each)")
parser.add_argument("--checkpoint", help="Number of steps between checkpoint saves")
parser.add_argument("--n_interior", help="Size of the interior of the lattice")
parser.add_argument("--padding", help="Size of the padding around the lattice")
parser.add_argument("--force_new", help="Can surpress checkpoint loading",
                    action="store_true")


def create_sim_and_submit(*args, **kwargs):
        sim = GlauberSimulatorFixIndices(*args, **kwargs)
        result = sim.run_single_glauber(verbose=True)
        return result


class Main:
    
    def __init__(self, *args, **kwargs) -> None:

        self.now = datetime.datetime.now()
        self.time_string = self.now.strftime("%m%d_%H-%M-%S")

        self.result_dir = RESULT_DIR  + self.time_string + "/"
        os.makedirs(self.result_dir, exist_ok=True)

        # create a new logging file for each instance
        pid = os.getpid()
        logger = logging.getLogger(__name__ + '.' +  str(pid))
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename=f'{self.result_dir}/log-main.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        self.logger = logger

        self.logger.info("running on " + socket.gethostname())

        if gitinfo.get_git_info() is not None:
            self.logger.info("git info: " + str(gitinfo.get_git_info())) 


    def check_for_checkpoints(self, params_dict):
        """If the last run in the results dir had the same parameters, will return that directory name and else None"""

        format = "%m%d_%H-%M-%S"

        contents = os.listdir(RESULT_DIR)
        runs = []
        for c in contents:
            if c == self.time_string:
                continue
            try:
                dt = datetime.datetime.strptime(c, format)  
            except ValueError as e:
                self.logger.info("found non-date directory: " + c)
                break
            files_of_run = os.listdir(RESULT_DIR + c)
            if "iterative-params.json" not in files_of_run:
                self.logger.info("found directory without iterative-params.json: " + c)
            else:
                runs.append(dt)
        
        if len(runs) == 0:
            self.logger.info("no runs found in directory " + self.result_dir)
            return None 
        
        runs.sort()
        last_run = runs[-1]
        last_run_str = datetime.datetime.strftime(last_run, format)
        self.logger.info("last run in directory " + self.result_dir + " was " + str(last_run))

        last_params = {}
        with open(self.result_dir + datetime.datetime.strftime(last_run, format) + "/iterative-params.json") as f:
            last_params = json.load(f)
        for key, value in params_dict.items():
            if value != last_params[key]:
                self.logger.info("last run had different parameters")
                return None
        
        self.logger.info("last run had same parameters, returning checkpoint directories")

        reps = os.listdir(self.result_dir + last_run_str + "/")
        reps = [last_run_str + '/' + x for x in reps if x.startswith("rep-")]

        return reps



    def run_traces(self, n_interior, padding, t, tol, iterations, p, results_dir, checkpoint_int, warmstarts):
        
        futures = []
        with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

            for i in range(iterations):
                future = executor.submit(create_sim_and_submit,
                                        padding = padding,
                                        n_interior = n_interior,
                                        p = p,
                                        t = t,
                                        tol = tol,
                                        results_dir = self.result_dir + 'rep-' + str(i),
                                        save_bitmaps_every =checkpoint_int,
                                        random_seed = i,
                                        checkpoint_file = warmstarts[i]
                                        )
                
                futures.append(future)

            for future in fts.as_completed(futures):
                result = future.result()

    def main(self, args = None):

        # if the script is run from the command line, args will be None, so the parser
        # will just use the supplied command line arguments
        
        np.random.seed(0)

        self.logger.info("starting " + self.time_string)
        self.logger.info("number of cores: " + str(mp.cpu_count()))

        args=parser.parse_args(args)

        self.logger.info("Started run for multiple traces with args" + str(args))
        
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

        P = int(args.p)

        params = {
            "n_interior": N_INTERIOR,
            "padding": PADDING,
            "t": T,
            "tol": TOL,
            "iterations": ITERATIONS,
            "p": P,
            "iterations": ITERATIONS,
        }  

        if not FORCE_NEW:
            checkpoint_runs = self.check_for_checkpoints(params)
        else:
            checkpoint_runs = None
            self.logger.info("Force new was set, so not loading checkpoints.")

        warmstarts = [None] * ITERATIONS

        if checkpoint_runs is not None and len(checkpoint_runs) > 0:
            self.logger.info("found checkpoint directories " + str(checkpoint_runs))
            params["warmstart_from"] = checkpoint_runs
            if len(checkpoint_runs) < ITERATIONS:
                self.logger.info("checkpoint does not have same number of runs, starting all from the same")
                warmstarts = [checkpoint_runs[0] + 'bitmap_results/iter-' + str(T) + '.bmp'] * ITERATIONS
            elif len(checkpoint_runs) == ITERATIONS:
                self.logger.info("checkpoint has same number of runs, starting each from its own")
                warmstarts = [RESULT_DIR + x + '/bitmap_results/iter-' + str(T) + '.bmp' for x in checkpoint_runs]
        
        
        self.run_traces(n_interior=N_INTERIOR, padding=PADDING, t=T, tol=TOL, 
                iterations=ITERATIONS, p=P, results_dir = self.result_dir, 
                checkpoint_int=CHECKPOINT_INTERVAL, warmstarts = warmstarts)

        params.update({"time_string": self.time_string})

        with open(self.result_dir + "iterative-params.json", "w") as f:
            json.dump(params, f)

        self.logger.info("finished " + self.time_string + ", took " + str(datetime.datetime.now() - self.now) + " to run")
        self.logger.info(f"saved results in directory {self.result_dir}")



if __name__ == "__main__":
    main = Main()
    main.main()

    
