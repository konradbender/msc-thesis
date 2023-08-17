import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from python.DataStructs.BitArrayMat import BitArrayMat
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def plot_bitmap(dir, iter):
    with (open(f"{dir}/params.json")) as f:
        params = json.load(f)


    matrix = BitArrayMat(params['n_outer'], params['n_outer'])
    matrix.load_from_file(dir + f"/iter-{iter}.bmp")
    np_mat = matrix.to_numpy()
    plt.imshow(np_mat)
    plt.title(f"Bitmap for iter={iter}")
    plt.savefig(dir + f"/iter-{iter}.png")

def plot_all_bitmaps_in_dir(dir):
    content = os.listdir(dir)

    iters = [x for x in content if x.endswith('.bmp')]
    futures = []
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        for i in iters:
            i = i.split('.')[0].split('-')[1]
            i = int(i)
            ft = executor.submit(plot_bitmap, dir, i)
            futures.append(ft)

        for future in as_completed(futures):
            pass


if __name__ == '__main__':
    dir = '/Users/konrad/code/school/msc-thesis/results/from-remote/0816_23-30-09/rep-0/bitmap_results'
    # plot_all_bitmaps_in_dir(dir)
    plot_bitmap("/Users/konrad/code/school/msc-thesis/results/from-remote/0816_23-30-09/rep-1/bitmap_results", 0)
    plot_bitmap("/Users/konrad/code/school/msc-thesis/results/from-remote/0816_23-30-09/rep-2/bitmap_results", 0)
    plot_bitmap("/Users/konrad/code/school/msc-thesis/results/from-remote/0816_23-30-09/rep-3/bitmap_results", 0)
