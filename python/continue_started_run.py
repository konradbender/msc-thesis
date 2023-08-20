from glauberFixIndices import GlauberSimulatorFixIndices
import logging
logging.basicConfig(level=logging.DEBUG)
import json

f_old_params = "/Users/konrad/code/school/msc-thesis/results/from-remote/0816_14-09-47/rep-0/simulation-params.json"
f_old_bitmap = "/Users/konrad/code/school/msc-thesis/results/from-remote/0816_14-09-47/rep-0/bitmap_results/iter-16820029.bmp"
f_results = "/Users/konrad/code/school/msc-thesis/results/from-remote/0816_14-09-47/rep-0/result-dict.json"

old_params = json.load(open(f_old_params, "r"))
old_results = json.load(open(f_results, "r"))

n_interior = old_params["n_interior"]
padding = old_params["padding"]
p = old_params["p"]
tol = old_params["tol"]
checkpoint_freq = old_params["save_bitmaps_every"]

old_steps = old_results["iterations"]
more_steps_to_do = 10e6

sim = GlauberSimulatorFixIndices(
    n_interior=int(n_interior),
    padding=int(padding),
    p=p,
    t=int(old_steps + more_steps_to_do),
    tol=1,
    results_dir="./results/continue_started_run",
    checkpoint_file=f_old_bitmap,
    random_seed=0,
    save_bitmaps_every=checkpoint_freq,
)

sim.run_single_glauber(verbose=True)

