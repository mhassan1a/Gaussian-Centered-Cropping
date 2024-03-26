#!/bin/bash
#SBATCH --partition=gpu4         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=64    # number of tasks per node
#SBATCH --time=3-00:00:00            # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --gres=gpu:1            # number of general-purpose GPUs
#SBATCH --mem=170G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

# Activate Conda (initialize Conda)
#eval "$(conda shell.bash hook)"
#conda activate nlp


#python ./src/cifer10_resnet18_mip_gcc.py #162139, 162142, 162172
#python ./src/cifer10_resnet18_mip_gccn.py #162140, 162143, 162176
#python ./src/cifer10_resnet18_mip_rc.py #162141
# Method: gcc(Gassian Center Crops with padding), gccr(Gaussian Center Crops Regularzed), rc
echo $SLURM_JOB_ID
python ./src/cifer10_resnet18_mip_gcc.py --method 'gcc' --crop_size 0.4 --std 0.001 0.01 0.1 0.5 1 1.5 2 3 5 100 200 --num_of_trials 4 --pretrain_epoch 150 --num_workers 4 --hidden_dim 128 --batchsize 512 --clf_epochs 100 