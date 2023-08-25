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

from glauber.glauberFixIndices import GlauberSimulatorFixIndices
from glauber.glauberDynIndices import GlauberSimDynIndices
from glauber.glauberTorus import GlauberFixedIndexTorus
from glauber.glauberTorus import GlauberDynIndexTorus

RESULT_DIR = "./results/"

parser=argparse.ArgumentParser()

parser.add_argument("--t", help="Number of iterations")
parser.add_argument("--p", help="Probability of +1 at initialization")
parser.add_argument("--n", help="Number of repetitions (how many times we run each)")
parser.add_argument("--checkpoint", help="Number of steps between checkpoint saves")
parser.add_argument("--n_interior", help="Size of the interior of the lattice")
parser.add_argument("--padding", help="Size of the padding around the lattice")
parser.add_argument("--force_new", help="Can surpress checkpoint loading",
                    action="store_true")
parser.add_argument("--tol", help="Tolerance to determine fixation")
parser.add_argument("--dynamic", help="if true, use dynamic indices", action="store_true")
parser.add_argument("--mixed", help="if true, use first fixed and then dynamci indices", action="store_true")
parser.add_argument("--fixed_steps", help="if mixed, how many steps to run fixed indices for")
parser.add_argument("--random_boundary", help="if set, will set boundary to random values", action="store_true")
parser.add_argument("--torus", help="if set, use a torus and not a square", action="store_true")


classes_square = {"fix": GlauberSimulatorFixIndices, "dyn": GlauberSimDynIndices}
classes_torus = {"fix": GlauberFixedIndexTorus, "dyn": GlauberDynIndexTorus}

classes = {"square": classes_square, "torus": classes_torus}


class Main:
    
    def __init__(self, arguments=None, *args, **kwargs) -> None:
        
        self.args = parser.parse_args(arguments )

        overwrite_result_dir = kwargs.get("result_dir", None)

        self.now = datetime.datetime.now()
        self.time_string = self.now.strftime("%m%d_%H-%M-%S")

        if overwrite_result_dir is not None:
            self.result_dir = overwrite_result_dir
        else:
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
        
        with open(os.path.join(self.result_dir, "run_multiple_for_traces.json"), "w") as f:
            json.dump(self.args.__dict__, f)
            
    def create_and_submit(self, structure, indexing, *args, **kwargs):
        sim = classes[structure][indexing](*args, **kwargs)
        self.logger.info("created simulator")
        result = sim.run_single_glauber(verbose=True)
        self.logger.info("simulator has finished")
        return result

    def create_fixed_and_then_dynamic(self, structure, fixed_steps, *args, **kwargs):
        fixed_args = kwargs.copy()
        fixed_args["t"] = fixed_steps
        sim1 = classes[structure]["fix"](*args, **fixed_args)
        self.logger.info("created fixed simulator for first steps")
        result1 = sim1.run_single_glauber(verbose=True)
        self.logger.info("first simulator has finished")
        
        checkpoint_file = os.path.join(sim1.results_dir, "bitmap_results", f"iter-{result1['iterations']}.bmp")
        kwargs["checkpoint_file"] = checkpoint_file
        sim2  = classes[structure]["dyn"](*args, **kwargs)
        self.logger.info("created second simulator for dynamic steps")
        result = sim2.run_single_glauber(verbose=True)
        self.logger.info("second simulator has finished")
        return result

    def find_last_run(self):
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
                continue
            files_of_run = os.listdir(RESULT_DIR + c)
            if "iterative-params.json" not in files_of_run:
                self.logger.info("found directory without iterative-params.json: " + c)
            else:
                runs.append(dt)
        
        if len(runs) == 0:
            self.logger.info("no runs found in directory " + RESULT_DIR)
            return None 
        
        runs.sort()
        last_run = runs[-1]
        
        return last_run

    def check_for_checkpoints(self, params_dict, last_run_str = None):
        """If the last run in the results dir had the same parameters, will return that directory name and else None"""

        format = "%m%d_%H-%M-%S"

        if last_run_str is None:
            last_run = self.find_last_run()
            if last_run is None:
                return None
            last_run_str = datetime.datetime.strftime(last_run, format)
            self.logger.info("last run in directory " + RESULT_DIR + " was " + str(last_run))
            last_run_str = os.path.join(RESULT_DIR, last_run_str)    

        last_params = {}

        with open(os.path.join(last_run_str, "iterative-params.json"), "r") as f:
            last_params = json.load(f)
        for key, value in params_dict.items():
            if key != "t" and value != last_params[key]:
                self.logger.info("last run had different parameters besides number of iterations")
                return None
        
        self.logger.info("last run had same parameters, returning checkpoint directories")

        reps = os.listdir(last_run_str)
        reps = [os.path.abspath(os.path.join(last_run_str, x)) for x in reps if x.startswith("rep-")]

        return reps



    def run_traces(self, n_interior, padding, t, tol, iterations, p, results_dir, checkpoint_int, warmstarts,
                   random_boundary=False):

        if self.args.dynamic:
            indexing = "dyn"
        elif self.args.mixed:
            indexing = "mixed"
        else:
            indexing = "fix"

        if self.args.torus:
            structure = "torus"
        else:
            structure = "square"

        futures = []
        with fts.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

            for i in range(iterations):
                run_args =  {"padding" : padding,
                        "n_interior" : n_interior,
                        "p" : p,
                        "t" : t,
                        "tol" : tol,
                        "results_dir" : self.result_dir + 'rep-' + str(i),
                        "save_bitmaps_every" :checkpoint_int,
                        "random_seed" : i,
                        "checkpoint_file" : warmstarts[i]
                        }
                if random_boundary:
                    run_args["boundary"] = "random"
                    
                if indexing == "mixed":
                    run_args["fixed_steps"] = int(self.args.fixed_steps)
                    future = executor.submit(self.create_fixed_and_then_dynamic, structure, **run_args)
                    self.logger.info("submitted mixed run")
                else:
                    future = executor.submit(self.create_and_submit, structure, indexing, **run_args)
                    self.logger.info("submitted fixed run")
                
                futures.append(future)
            
            return_value = 0

            for future in fts.as_completed(futures):
                try:
                    result = future.result()
                    self.logger.info("finished run")
                except Exception as e:
                    self.logger.error(f"Some run caused an error: {e}")
                    return_value = 1
        return return_value

    def main(self, last_run=None):

        # if the script is run from the command line, args will be None, so the parser
        # will just use the supplied command line arguments
        
        np.random.seed(0)

        self.logger.info("starting " + self.time_string)
        self.logger.info("number of cores: " + str(mp.cpu_count()))

        self.logger.info("Started run for multiple traces with args" + str(self.args))
        
        # number of time steps for each glauber dynamics iteration
        T = int(self.args.t)

        # number of times we run glauber dynamcis for each probability
        ITERATIONS = int(self.args.n)

        # dimension of the lattice B
        N_INTERIOR = int(self.args.n_interior)
        # dimension of the padding around B
        PADDING = int(self.args.padding)

        CHECKPOINT_INTERVAL = int(self.args.checkpoint)

        FORCE_NEW = bool(self.args.force_new)

        
        # tolerance for fixation
        if self.args.tol is None:
            self.logger.info("tolerance was not set, using default 1")
            TOL = 1
        else:
            TOL = float(self.args.tol)

        P = float(self.args.p)

        params = {
            "n_interior": N_INTERIOR,
            "padding": PADDING,
            "t": T,
            "tol": TOL,
            "iterations": ITERATIONS,
            "p": P,
            "iterations": ITERATIONS,
            "torus": self.args.torus,
        }  

        if not FORCE_NEW:
            checkpoint_runs = self.check_for_checkpoints(params, last_run_str=last_run)
        else:
            checkpoint_runs = None
            self.logger.info("Force new was set, so not loading checkpoints.")

        warmstarts = [None] * ITERATIONS

        if checkpoint_runs is not None and len(checkpoint_runs) > 0:
            self.logger.info("found checkpoint directories " + str(checkpoint_runs))
            params["warmstart_from"] = checkpoint_runs
            if len(checkpoint_runs) < ITERATIONS:
                self.logger.info("checkpoint does not have same number of runs, starting all from the same")
                dir = os.path.join(checkpoint_runs[0], 'bitmap_results')
                bitmaps = os.listdir(dir)
                bitmaps = [x for x in bitmaps if x.endswith('.bmp')]
                bitmaps.sort()
                last_bitmap = bitmaps[-1]
                warmstarts = [os.path.join(dir, last_bitmap)] * ITERATIONS

            elif len(checkpoint_runs) == ITERATIONS:
                self.logger.info("checkpoint has same number of runs, starting each from its own")
                warmstarts = []
                for run in checkpoint_runs:
                    dir = os.path.join(run ,'bitmap_results')
                    files = os.listdir(dir)
                    bitmaps = [x for x in files if x.endswith('.bmp')]
                    bitmaps = [(int(x.split('-')[1].split('.')[0]), x) for x in bitmaps]
                    bitmaps.sort()
                    last_bitmap = bitmaps[-1][1]
                    warmstarts.append(os.path.join(dir, last_bitmap))
        
        
        exit_code = self.run_traces(n_interior=N_INTERIOR, padding=PADDING, t=T, tol=TOL, 
                iterations=ITERATIONS, p=P, results_dir = self.result_dir, 
                checkpoint_int=CHECKPOINT_INTERVAL, warmstarts = warmstarts,
                random_boundary=self.args.random_boundary)

        params.update({"time_string": self.time_string})

        if self.args.random_boundary:
            params.update({"boundary": "random"})

        with open(self.result_dir + "iterative-params.json", "w") as f:
            json.dump(params, f)

        self.logger.info("finished " + self.time_string + ", took " + str(datetime.datetime.now() - self.now) + " to run")
        self.logger.info(f"saved results in directory {self.result_dir}")

        return exit_code



if __name__ == "__main__":
    main = Main()
    result = main.main()
    sys.exit(result)

    
