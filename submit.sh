#!/bin/bash
#SBATCH --job redo-the-500-20M
#SBATCH --mem 16G
#SBATCH --partition=standard-cpu

# THE FOLLOWING TWO MUST BE MANUALLY ALIGNED -  ONCE ORE MORE THAN ITERS
N_INT=4
#SBATCH --cpus-per-task 5

#SBATCH --mail-user=kobender@stats.ox.ac.uk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

echo “I am job number ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} running on the computer ${HOSTNAME}”

# activate venv
source thesis-env-2/bin/activate

T=20000000
N=4
CHECKPOINT=1000000

PADDING=10

# run the script
python3 python/run_multiple_for_traces.py --t=$T --n=$N --checkpoint=$CHECKPOINT --n_int=$N_INT --padding=$PADDING 