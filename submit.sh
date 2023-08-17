#!/bin/bash
#SBATCH --job testing
#SBATCH --mem 16G
#SBATCH --mail-user=kobender@stats.ox.ac.uk
#SBATCH --partition=standard-cpu

echo “I am job number ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} running on the computer ${HOSTNAME}”

# activate venv
source thesis-env-2/bin/activate

T=2000
N=4
CHECKPOINT=10000

# THE FOLLOWING TWO MUST BE MANUALLY ALIGNED
N_INT=4
#SBATCH --cpus-per-task 5

PADDING=10

# run the script
python3 python/run_multiple_for_traces.py --t=$T --n=$N --checkpoint=$CHECKPOINT --n_int=$N_INT --padding=$PADDING 