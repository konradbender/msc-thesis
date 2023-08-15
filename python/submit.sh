#!/bin/bash
#SBATCH --job testing
#SBATCH --cpus-per-task 5
#SBATCH --mem 5G
#SBATCH --mail-user=kobender@stats.ox.ac.uk
#SBATCH --partition=standard-cpu

echo “I am job number ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} running on the computer ${HOSTNAME}”

source kobender-msc-thesis/bin/activate

# activate venv
python3 test_glauber.py