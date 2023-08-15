#!/bin/bash
#SBATCH --job testing
#SBATCH --cpus-per-task 5
#SBATCH --mem 16G
#SBATCH --mail-user=kobender@stats.ox.ac.uk
#SBATCH --partition=standard-cpu

echo “I am job number ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} running on the computer ${HOSTNAME}”

# activate venv
source thesis-env-2/bin/activate

# run the script
python3 python/test_glauber.py