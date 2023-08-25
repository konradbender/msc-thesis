#!/bin/bash
#SBATCH --job redo-the-500-20M
#SBATCH --mem 16G
#SBATCH --partition=standard-cpu
#SBATCH --export=ALL
#SBATCH --mail-type=all

# THE FOLLOWING TWO MUST BE MANUALLY ALIGNED -  ONCE ORE MORE THAN ITERS
N=7
#SBATCH --cpus-per-task 8

echo “I am job running on the computer ${HOSTNAME}”

# activate venv
source /vols/teaching/msc-projects/2022-2023/kobender/msc-thesis/thesis-env-2/bin/activate
which python
echo $PATH

T=20000000 # time steps
N_INT=100 # n_interior, so the size of the grid
CHECKPOINT=100000 # steps between model checkpoints

P=0.505

# run the script
python3 python/run_multiple_for_traces.py --t=$T --n=$N --checkpoint=$CHECKPOINT \
 --n_int=$N_INT --padding=0 --p=$P --mixed --fixed_steps=500000 --random_boundary --torus