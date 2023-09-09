import plot_bitmaps as pm
import plot_traces
import make_gif as mg
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')   

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def process_run(run_dir):
    reps = [int(x.split('-')[1]) for x in os.listdir(run_dir) if x.startswith('rep-')]
    if len(reps) == 0:
        return
    for i in range(0, max(reps) + 1):
        bitmap_path = os.path.join(run_dir, f"rep-{i}", "bitmap_results")
        if not os.path.isdir(bitmap_path):
            continue
        try:
            pm.plot_all_bitmaps_in_dir(bitmap_path)
            mg.make_gif(bitmap_path)
            pass
        except Exception as e:
            print(e)
        print(f"For rep {i} done")
    
    try:        
        plot_traces.one_plot_for_all(run_dir)
    except Exception as e:
        print(e)
     

def process_stem(stem, force_new=False):
    results = os.listdir(stem)
    for r in results:
        if r.startswith('.'):
            print(f"Skipping {r}")
            continue
        if "traces-all.pdf" in os.listdir(os.path.join(stem, r)) and not force_new:
            print(f"Already done {r}")
            continue
        print(f"Starting with {r}")
        run_dir = os.path.join(stem, r)
        process_run(run_dir)
        print(f"Done with {r}")
    
    
    
if __name__ == '__main__':
    plt.ioff()
    process_stem("/Users/konrad/code/school/msc-thesis/results/from-remote", force_new=True)
    
    