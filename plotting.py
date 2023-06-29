import json
import matplotlib.pyplot as plt
import pandas as pd
import os


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


if __name__ == '__main__':
    file = 'results/results-100k-0623_1445.json'
    plot_fixation_rates(file)