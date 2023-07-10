import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def plot_fixation_rates(file):
    os.makedirs('plots', exist_ok=True)

    with open(file, 'r') as f:
        data = json.load(f)

    setup = data[0]
    results = data[1:]

    n_outer = setup['n_outer']
    n_inner = setup['n_inner']
    t = setup['t']
    tol = setup['tol']
    iterations = setup['iterations']
    time_string = setup['time_string']

    df = pd.DataFrame(data = results)
    df = df.sort_values(by=['p'], ascending=True)


    fig, ax = plt.subplots()
    plt.plot(df.p, df.fixation_rate, 'o')

    ax.set_title(f"Fixation Rates for t={t}, N={iterations}, tol={tol}, n={n_outer}, m={n_inner}")
    plt.savefig(f"plots/fixation_rates-{time_string}.png")

def plot_trace(file):
    with open(file, 'r') as f:
        data = json.load(f)

   
    n_outer = data['n_outer']
    n_inner = data['n_inner']
    t = data['t']
    tol = data['tol']
    iterations = data['iterations']
    p = data['p']
    time_string = data['time_string']

    traces = np.array(data['traces'])
    fixations = np.array(data['fixations'])

    fig, ax = plt.subplots()
    for i in range(iterations):
        plt.plot(np.arange(t), traces[i][traces[i] != -1])
    ax.set_title(f"Traces for t={t}, N={iterations}," + \
                 f"tol={tol}, n={n_outer}, m={n_inner}, p={p}")
    plt.savefig(f"plots/trace-{time_string}.png")
    plt.show()


if __name__ == '__main__':
    file = 'trace-results/results-1M-0707_1340.json'
    plot_trace(file)