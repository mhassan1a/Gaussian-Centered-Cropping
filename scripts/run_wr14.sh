python ./src/train_1gpu.py \
    --method gcc \
    --crop_size 0.2 \
    --std  3 \
    --num_of_trials 1 \
    --pretrain_epoch 200 \
    --num_workers 3 \
    --hidden_dim 128 \
    --batchsize 512 \
    --clf_epochs 100 \
    --dataset 'Cifar10' \
    --model 'Proto18' \
    --adaptive_center True\
    --job_id 0\
    --min_max 0.25 0.75




#options Dataset: Cifar10, Cifar100, TinyImageNet, Imagenet64
#options Model: Proto18, Proto34
#options Method: rc (pytorch random crop), gcc, gccr
# python -m torch.distributed.launch --nproc_per_node=4 main_script.py