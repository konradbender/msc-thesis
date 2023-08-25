import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from python.glauber.DataStructs.BitArrayMat import BitArrayMat


def plot_trace(dir, data):
    
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

def get_data(dir):
    with open(dir + '/result-dict.json', 'r') as f:
                result = json.load(f)
    
    with open(dir + '/simulation-params.json', 'r') as f:
        params = json.load(f)

    data = {}
    data.update(result)
    data.update(params)

    return data


def one_plot_per_rep(stem):
    content = os.listdir(stem)
    for dir in content:
        if dir.startswith('rep-'):
            data = get_data(stem + '/' + dir)
            plot_trace(stem + '/' + dir, data)


def one_plot_for_all(stem):
    content = os.listdir(stem)
    traces = {}
    with open(stem + '/iterative-params.json', 'r') as f:
        params = json.load(f)

    for dir in content:
        if dir.startswith('rep-'):
            try:
                data = get_data(stem + '/' + dir)
            except:
                continue
            
            if "vector" in data.keys() and "iterations" in data.keys():
                traces[dir] = np.array(data["vector"])

    fig, ax = plt.subplots()
    
    for dir, trace in traces.items():
        good_data = trace[trace != -1]
        plt.plot(np.arange(good_data.shape[0]), good_data, label=dir)
    
    ax.set_title(f"Traces for t={params['t']}," + \
                f"tol={params['tol']}, m={params['n_interior']}, p={params['p']}")
    
    plt.legend(loc="upper left")
    
    plt.savefig(f"{stem}/traces-all.png")

            



if __name__ == '__main__':
    # one_plot_per_rep("/Users/konrad/code/school/msc-thesis/results/from-remote/0817_11-52-28")
    one_plot_for_all("/Users/konrad/code/school/msc-thesis/results/from-remote/0825_11-13-21")

    