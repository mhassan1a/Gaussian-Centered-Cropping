#!/bin/bash
#SBATCH --partition=gpu         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=40     # number of tasks per node
#SBATCH --time=10:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --gres=gpu:1             # number of general-purpose GPUs
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR


echo "Starting Job"
python ./src/cifer10_resnet18.py