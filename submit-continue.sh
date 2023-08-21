#!/bin/bash
#SBATCH --job continue_old_runs
#SBATCH --mem 16G
#SBATCH --partition=standard-cpu
#SBATCH --export=ALL
#SBATCH --mail-type=all

#SBATCH --cpus-per-task 7

echo “I am job running on the computer ${HOSTNAME}”

# activate venv
source /vols/teaching/msc-projects/2022-2023/kobender/msc-thesis/thesis-env-2/bin/activate
which python
echo $PATH

STEM_PATH="/vols/teaching/msc-projects/2022-2023/kobender/msc-thesis/results/0819_20-48-20" 

# run the script
python3 python/continue_started_run.py --stem=$STEM_PATH