import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches as mpatches
import os
import numpy as np
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from python.glauber.DataStructs.BitArrayMat import BitArrayMat
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

CMAP = 'bwr'

values = [0,1]
    
labels = {0:'-1', 1:'+1'}
    

def plot_bitmap(dir, iter, save = True):
    with (open(f"{dir}/params.json")) as f:
        params = json.load(f)
        
    matrix = BitArrayMat(params['n_outer'], params['n_outer'])
    matrix.load_from_file(dir + f"/iter-{iter}.bmp")
    np_mat = matrix.to_numpy()
    
    im = plt.matshow(np_mat, cmap="bwr", vmin=0, vmax=1)
    
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], 
                                label=labels[values[i]])  for i in range(len(values)) ]

    plt.legend(handles=patches, loc='lower center',bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.title("Bitmap for Iter. {:,.0f}".format(iter))
    plt.tight_layout()
    if save:
        plt.savefig(dir + f"/iter-{iter}.png", dpi=75, bbox_inches='tight')
        plt.close()
    return im
    
    
def plot_three_bitmaps(dir_stem, rep, iters, save = True, title = True):
    dir = os.path.join(dir_stem, f"rep-{rep}", "bitmap_results")
    fig, ax = plt.subplots(1,3, figsize=(11,3.5))
    for i, iter in enumerate(iters):
        with (open(f"{dir}/params.json")) as f:
            params = json.load(f)
            
        matrix = BitArrayMat(params['n_outer'], params['n_outer'])
        try:
            matrix.load_from_file(dir + f"/iter-{iter}.bmp")
            np_mat = matrix.to_numpy()
        except FileNotFoundError:
            UserWarning(f"This file does not exist: {dir + f'/iter-{iter}.bmp'}")
            np_mat = np.full((params['n_outer'], params['n_outer']), np.nan)        
    
        im = ax[i].matshow(np_mat, cmap="bwr", vmin=0, vmax=1)
        
        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], 
                                    label=labels[values[i]])  for i in range(len(values)) ]
        
        ax[i].set_title('Iter. {:,.0f}'.format(iter))
    
    if title:
        fig.suptitle(f"Bitmaps for different iterations of repetition 'rep-{rep}' of Experiment '{os.path.basename(dir_stem)}'")
        fig.legend(handles=patches, ncols=2)
    else:
        fig.legend(handles=patches, ncols=2, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(os.path.dirname(dir), f"three-bitmaps-rep-{rep}.pdf"))
    return fig, ax

    

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

    