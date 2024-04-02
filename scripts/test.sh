#!/bin/bash
#SBATCH --partition=gpu4         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=64       # number of tasks per node
#SBATCH --time=3-0:00:00         # total runtime of job allocation
#SBATCH --gres=gpu:1             # number of general-purpose GPUs
#SBATCH --mem=170G               # memory per node in MB
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo 'Job ID: ' $SLURM_JOB_ID$SLURM_ARRAY_TASK_ID
# Define an array of parameters
methods=('gcc' 'gccr' 'gcc' 'gccr' 'gcc' 'gccr' 'gcc' 'gccr')
crop_sizes=(0.2 0.2 0.4 0.4 0.6 0.6 0.8 0.8)

# Extract parameters for this job
method=${methods[$SLURM_ARRAY_TASK_ID]}
crop_size=${crop_sizes[$SLURM_ARRAY_TASK_ID]}

# Execute your command with the extracted parameters
python ./src/cifer10_resnet18_mip_gcc.py \
    --method test \
    --crop_size $crop_size \
    --std 0.001  \
    --num_of_trials 4 \
    --pretrain_epoch 0 \
    --num_workers 4 \
    --hidden_dim 128 \
    --batchsize 512 \
    --clf_epochs 1 \
    --dataset 'Cifar10' \
    --model 'Proto18' \
    --adaptive_center True\
    --job_id $SLURM_JOB_ID$SLURM_ARRAY_TASK_ID

# Instructions for running the script:
# - To run this script, use the following command:
#   sbatch --array=0-7 scripts/slurm.sh
# - The array index will be passed to the script as SLURM_ARRAY_TASK_ID
# - The script will then extract the corresponding parameters from the arrays and execute the command with these parameters
