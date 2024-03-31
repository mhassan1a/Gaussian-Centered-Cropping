#!/bin/bash
#SBATCH --partition=any         # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=64    # number of tasks per node
#SBATCH --time=3-0:00:00            # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --gres=gpu:1            # number of general-purpose GPUs
#SBATCH --mem=170G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"


# Method: gcc(Gassian Center Crops with padding), gccr(Gaussian Center Crops Regularzed), rc (Random Crops) , gccr and gccr with adaptive centers
# DATASETS for pretraining Cifar10, Cifar100, Imagenet64
# Models: Proto18, //? Proto50, SimCLR, MoCo, BYOL, SimSiam

echo $SLURM_JOB_ID
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gcc' --crop_size 0.6 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gccr' --crop_size 0.6 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gcc' --crop_size 0.4 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gccr' --crop_size 0.4 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gcc' --crop_size 0.8 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'
#python ./src/cifer10_resnet18_mip_gcc.py --method 'gccr' --crop_size 0.8 --std 0.001 0.01 0.1 0.5 1.0 1.5 2.0 3.0 5.0 100.0 200.0   --num_of_trials 4 --pretrain_epoch 300 --num_workers 4 --hidden_dim 256 --batchsize 512 --clf_epochs 100 --dataset 'Cifar100' --model 'Proto18'

