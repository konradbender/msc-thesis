#!/bin/bash
#SBATCH --job run-unit-tests
#SBATCH --mem 16G
#SBATCH --partition=standard-cpu
#SBATCH --export=ALL
#SBATCH --mail-type=all

# THE FOLLOWING TWO MUST BE MANUALLY ALIGNED -  ONCE ORE MORE THAN ITERS

#SBATCH --cpus-per-task 5

echo “I am job number running on the computer ${HOSTNAME}”

# activate venv
source /vols/teaching/msc-projects/2022-2023/kobender/msc-thesis/thesis-env-2/bin/activate
which python
echo $PATH

python3 pytest --durations=0 -c pyproject.toml