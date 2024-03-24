#!/bin/bash
#SBATCH --partition=gpu4         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=128    # number of tasks per node
#SBATCH --time=3-00:00:00            # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --gres=gpu:1            # number of general-purpose GPUs
#SBATCH --mem=125G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

# Activate Conda (initialize Conda)
#eval "$(conda shell.bash hook)"
#conda activate nlp


#python ./src/cifer10_resnet18_mip_gcc.py #162139
#python ./src/cifer10_resnet18_mip_gccn.py #162140
#python ./src/cifer10_resnet18_mip_rc.py #162141
