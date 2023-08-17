import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from python.DataStructs.BitArrayMat import BitArrayMat


def plot_trace(dir):
    
    with open(dir + '/result-dict.json', 'r') as f:
        result = json.load(f)
    
    with open(dir + '/simulation-params.json', 'r') as f:
        params = json.load(f)

    data = {}
    data.update(result)
    data.update(params)

    
    padding = data['padding']
    n_inner = data['n_interior']

    n_outer = padding + n_inner

    t = data['t']
    tol = data['tol']
    iterations = data['iterations']
    p = data['p']

    trace = np.array(data['vector'])

    fig, ax = plt.subplots()

    good_data = trace[trace != -1]

    plt.plot(np.arange(good_data.shape[0]), good_data)
    ax.set_title(f"Traces for t={t}," + \
                 f"tol={tol}, n={n_outer}, m={n_inner}, p={p}")
    plt.savefig(f"{dir}/trace.png")

if __name__ == '__main__':
    result_dir = "/Users/konrad/code/school/msc-thesis/results/from-remote/0817_11-52-28"
    content = os.listdir(result_dir)
    for dir in content:
        if (dir.startswith('rep-')):
            plot_trace(result_dir + '/' + dir)