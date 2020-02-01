#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/dshahaf/omribloch/env/snake/snake/bin/activate.csh
module load tensorflow

python3 <YOUR_FOLDER>/Snake.py -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(lr=0.001);MyPolicy(lr=0.001)" -D 5000 -s 1000 -l "<YOUR_LOG_PATH>" -r 0 -plt 0.01 -pat 0.005 -pit 60
