import json
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
from matplotlib import patches as mpatches
import os
import numpy as np
from python.glauber.DataStructs.BitArrayMat import BitArrayMat
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def plot_bitmap(dir, iter):
    with (open(f"{dir}/params.json")) as f:
        params = json.load(f)
        
    matrix = BitArrayMat(params['n_outer'], params['n_outer'])
    matrix.load_from_file(dir + f"/iter-{iter}.bmp")
    np_mat = matrix.to_numpy()
    
    values = [0,1]
    
    labels = {0:'-1', 1:'1'}
    
    im = plt.imshow(np_mat, cmap="bwr", vmin=0, vmax=1)
    
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], 
                               label=labels[values[i]])  for i in range(len(values)) ]

    plt.legend(handles=patches, loc='lower center',bbox_to_anchor=(0.5, -0.2), ncol=2)

    plt.title(f"Bitmap for iter={iter}")
    plt.tight_layout()
    plt.savefig(dir + f"/iter-{iter}.png", dpi=75)

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

    