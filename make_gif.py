import imageio.v3 as iio
import matplotlib.pyplot as plt
from pygifsicle import optimize
import os
import numpy as np
import os


def make_gif(plot_dir):
    contents = os.listdir(plot_dir)
    plots = [x for x in contents if x.endswith('.png')]
    plots = [(int(x.split('.')[0].split('-')[1]), x) for x in plots]
    plots.sort()
    frames = np.stack([iio.imread(os.path.join(plot_dir, x[1])) for x in plots], axis=0)
    
    gif_path = os.path.join(os.path.dirname(plot_dir), "animation.gif")
    iio.imwrite(gif_path, frames)
    optimize(gif_path)
    
if __name__ == '__main__':
    make_gif("/Users/konrad/code/school/msc-thesis/results/from-remote/0825_21-37-01/rep-3/bitmap_results")
    
    
