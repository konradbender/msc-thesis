# Glauber Dynamics on Percolated Lattices

This repo holds the code for my Master Thesis on Glauber Dynamics [(Wikipedia)](https://en.wikipedia.org/wiki/Glauber_dynamics). The code for simulation is in [`python/`](./python/) and `/c++`. Plotting codes are in [`plotting/`](./plotting/). 

Results are in [`thesis-results/`](./thesis-results/). There are three directories for the three different results we ran. For each of those three, you find `traces-all.png` which shows the development of the share of $+1$ vertices in the grid. For each experiment, we ran the simulations 10 times, and for each run you'll find a directory `/rep-k` where the details of the k-th run are stored. Also, the animations of the grids over time are visible in `animation.gif`.

For example, below is one of those animations. It won't loop on GitHub, so you'll have to re-load the page to see it again.

![](./thesis-results/0826_19-14-32-torus/rep-3/animation.gif)