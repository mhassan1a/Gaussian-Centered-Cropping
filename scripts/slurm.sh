#!/bin/bash
#SBATCH --partition=gpu4       # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=32       # number of tasks per node
#SBATCH --time=0-24:00:00         # total runtime of job allocation
#SBATCH --gres=gpu:1             # number of general-purpose GPUs
#SBATCH --mem=150G               # memory per node in MB
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo 'Job ID: ' $SLURM_JOB_ID$SLURM_ARRAY_TASK_ID
# Define an array of parameters
methods=('gcc' 'gcc' 'gcc' )
min_max_list=('0.25 0.75' '0.4 0.6' '0.25 0.75')
adaptive_centers=(False False False)
crop_sizes=(0.2 0.4 0.6)
stds=(0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0 10 50 100.0 200.0)
# Extract parameters for this job
method=${methods[$SLURM_ARRAY_TASK_ID]}
crop_size=${crop_sizes[$SLURM_ARRAY_TASK_ID]}
adaptive_center=${adaptive_centers[$SLURM_ARRAY_TASK_ID]}
min_max=${min_max_list[$SLURM_ARRAY_TASK_ID]}
# Execute your command with the extracted parameters
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_NUM_THREADS=1
python ./src/train_1gpu.py \
    --method $method \
    --crop_size $crop_size \
    --std  3 \
    --num_of_trials 3 \
    --pretrain_epoch 200 \
    --num_workers 3 \
    --hidden_dim 128 \
    --batchsize 512 \
    --clf_epochs 100 \
    --dataset 'TinyImageNet' \
    --model 'Proto18' \
    --adaptive_center $adaptive_center\
    --job_id $SLURM_JOB_ID$SLURM_ARRAY_TASK_ID\
    --min_max $min_max



# Instructions for running the script:
# - To run this script, use the following command:
#   sbatch --array=0-7 scripts/slurm.sh
# - The array index will be passed to the script as SLURM_ARRAY_TASK_ID
# - The script will then extract the corresponding parameters from the arrays and execute the command with these parameters

#options Dataset: Cifar10, Cifar100, TinyImageNet, Imagenet64
#options Model: Proto18, Proto34
#options Method: rc (pytorch random crop), gcc, gccr
# python -m torch.distributed.launch --nproc_per_node=4 main_script.py