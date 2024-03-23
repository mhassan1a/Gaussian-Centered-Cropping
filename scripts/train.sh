#!/bin/bash
#SBATCH --partition=gpu4         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=128    # number of tasks per node
#SBATCH --time=0-08:00:00            # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --gres=gpu:1            # number of general-purpose GPUs
#SBATCH --mem=300G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

# Activate Conda (initialize Conda)
#eval "$(conda shell.bash hook)"
#conda activate nlp

#python ./src/cifer10_resnet18.py
#python ./src/cifer10_resnet18dp.py

python ./src/cifer10_resnet18_mip.py