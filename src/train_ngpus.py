import os
gpus = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])

from misc import EarlyStopper, convert_to_normal_dict
from datasets import TwoViewCifar10, TwoViewCifar100, TwoViewImagenet64, TinyImageNetDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from cropping import GaussianCrops
from models import Proto18,Proto34, Classifier
import numpy as np
import torch
import torch.nn as nn
from lightly.loss import NegativeCosineSimilarity, ntx_ent_loss
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms
import json
from datetime import datetime
import multiprocessing as multiprocessing
from multiprocessing import Manager as mp_manager
from mlp import NestablePool as MyPool
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from torch import distributed as dist


def train_epoch(epoch,model, dataloader, optimizer, scheduler, criterion, device):
    #one epoch of training
    dataloader.sampler.set_epoch(epoch)
    model.train()
    losses = []
    for i, (view1, view2, target) in enumerate(dataloader):
        view1, view2, target = view1.to(device), view2.to(device), target.to(device)
        optimizer.zero_grad()
        output1 = model(view1)
        output2 = model(view2)
        loss = criterion(output1, output2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    return np.mean(losses)

def distributed_dataloader(dataset, num_workers, batch_size=512, local_rank=0, world_size=4):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, 
                                                              num_replicas=world_size, 
                                                              rank=local_rank)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            sampler=sampler, 
                           )
    return dataloader

def distributed_model(model, local_rank):
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model

def pretraining(MODEL,DATASET,max_epochs=100, batch_size=512, num_workers=4, cropping=None, transform=None, hidden_dim=256):
    print(f"GPU Availble:{torch.cuda.is_available()}")
    print('Pre-Traing Starts ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    world_size = torch.cuda.device_count()
    print(f"World Size: {world_size}")

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"Local Rank: {local_rank}")
    
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=world_size, 
                            rank=local_rank)
    
    torch.cuda.set_device(local_rank)
    
    if DATASET.__name__ == 'TwoViewImagenet64':
        root = './data/Imagenet64_train'
    elif DATASET.__name__ == 'TinyImageNetDataset':
        root = './data/tiny-imagenet-200'
    else:
        root = './data'
        
    train_dataset = DATASET(root=root, train=True, download=True, cropping=cropping, transform=transform)
    train_loader = distributed_dataloader(train_dataset, num_workers, batch_size, local_rank, world_size)
    model = MODEL(hidden_dim = hidden_dim).cuda(local_rank)
    model = distributed_model(model, local_rank)
    optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs )
    criterion = ntx_ent_loss.NTXentLoss(temperature = 0.1, memory_bank_size=0)
    train_progress = tqdm(range(max_epochs), desc='pre training', unit='epoch')
    pretrain_loss = []
    for epoch in train_progress:
        loss = train_epoch(epoch, model, train_loader, optimizer, scheduler, criterion, device)
        pretrain_loss.append(loss)
        train_progress.set_postfix(loss=loss)
    return model.module, pretrain_loss


def addclassifier(model, num_classes):
    for name , param in model.resnet.named_parameters():
        param.requires_grad = False

    num_features = model.resnet.fc.in_features
    classifier = Classifier(num_features, hidden_dim=512, num_classes=num_classes)
    model.resnet.fc = classifier
    return model

def train_classifier(model, max_epochs=100, earlystop_patience=10, num_workers=1):
    print('Training Classifier...')
 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
                mean = (0.4914, 0.4822, 0.4465),
                std = (0.2023, 0.1994, 0.2010)
            )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
                std = (0.2023, 0.1994, 0.2010)
            ),
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers =num_workers )
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers =num_workers )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers =num_workers )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9
            )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    early_stop = EarlyStopper(patience= earlystop_patience, min_delta=0.001)
    num_epochs = max_epochs
    
    
    for epoch in range(1,num_epochs+1):
        
        model.train()  
        running_loss = 0.0 
        correct_predictions = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        train_accuracy = correct_predictions / len(train_dataset)
        
        # Validation 
        model.eval()  
        val_running_loss = 0.0
        val_correct_predictions = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct_predictions += (val_predicted == val_labels).sum().item()
                val_running_loss += val_loss.item() * val_images.size(0)
        # Calculate average validation loss and accuracy
        val_loss = val_running_loss / len(val_dataset)
        val_accuracy = val_correct_predictions / len(val_dataset)
        # early stopping
        if early_stop.early_stop(val_loss):
            break
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total      
    return test_accuracy , val_accuracy, train_accuracy


def run_trial(MODEL, DATASET, method, crop_size, adaptive_center,min_max , std, trial_num, num_workers, pretrain_epoch, batchsize, hidden_dim, clf_epochs):
    view_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    
    crop_percentage = crop_size
    seed = None
    std_scale = std
    pad = True
    reg = False
    if method == 'gccr': # Gaussian Cropping with Regularisation
        pad = False
        reg = True

    crop = GaussianCrops(crop_percentage=crop_percentage,
                         seed=seed,
                         std_scale=std_scale,
                         padding=pad,
                         regularised_crop=reg,
                         adaptive_center=adaptive_center,
                         min_max=min_max,)
    
    if method == 'rc': # Random Cropping 
        crop_dim = int(32 * np.sqrt(crop_size))
        crop = transforms.RandomCrop(crop_dim, padding=0)
        
    model, pretrain_loss = pretraining(MODEL= MODEL,
                                       DATASET=DATASET,
                                       max_epochs=pretrain_epoch,
                                       batch_size=batchsize,
                                       num_workers=num_workers,
                                       cropping=crop,
                                       transform=view_transform,
                                       hidden_dim=hidden_dim)

    model = addclassifier(model, 10)
    test_accuracy, val_accuracy, train_accuracy = train_classifier(model=model,
                                                                   max_epochs=clf_epochs,
                                                                   earlystop_patience=15,
                                                                   num_workers=num_workers)
    print()
    print(f"Method: {method}, Crop Size: {crop_size}, std: {std}, Trial: {trial_num}, Test Accuracy: {test_accuracy}, Val Accuracy: {val_accuracy}, Train Accuracy: {train_accuracy}")
    return test_accuracy, val_accuracy, train_accuracy


argparser = argparse.ArgumentParser()
argparser.add_argument('--method', type=str, default='gccr',required=True)
argparser.add_argument('--crop_size', type=float, nargs='+',required=True)
argparser.add_argument('--std', type=float, nargs='+')
argparser.add_argument('--num_of_trials', type=int, default=1,required=True)
argparser.add_argument('--pretrain_epoch', type=int, default=150,required=True)
argparser.add_argument('--num_workers', type=int, default=4,required=True)
argparser.add_argument('--hidden_dim', type=int, default=256,required=True)
argparser.add_argument('--batchsize', type=int, default=512,required=True)
argparser.add_argument('--clf_epochs', type=int, default=100,required=True)
argparser.add_argument('--dataset', type=str, default='Cifar10', required=False, help='Pretraining dataset')
argparser.add_argument('--model', type=str, default='Proto18', required=False)
argparser.add_argument('--adaptive_center', type=str, default='False', required=False)
argparser.add_argument('--job_id', type=str, default=None, required=False)
argparser.add_argument('--min_max', type=float, nargs='+', default=[0.25,0.75], required=False)
args = argparser.parse_args()

Dataset_lookup = {
                  'Cifar10': TwoViewCifar10, 
                  'Cifar100': TwoViewCifar100, 
                  'Imagenet64': TwoViewImagenet64,
                  'TinyImageNet': TinyImageNetDataset
                  }

Model_lookup = {
                'Proto18': Proto18,
                'Proto34': Proto34,
                }


print(args)
timestamp = datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
print('Timestamp:', timestamp)

if __name__ == '__main__':
    # config
    pretrain_epoch = args.pretrain_epoch
    num_workers = args.num_workers
    hidden_dim = args.hidden_dim
    batchsize = args.batchsize
    clf_epochs = args.clf_epochs
    methods = [args.method]
    crop_sizes = args.crop_size
    num_of_trials = args.num_of_trials
    job_id = args.job_id
    stds = args.std
    adaptive_center = True if args.adaptive_center=='True' else False
    MODEL = Model_lookup[args.model]
    DATASET = Dataset_lookup[args.dataset]
    min_max = args.min_max          

    # Create multiple processes to run multiple trials in parallel
    results = {}
    for method in methods:
        results[method] = {}
        for crop_size in crop_sizes:
            results[method][crop_size] = {}
            for std in stds:
                results[method][crop_size][std] = []
                # Parallelize trials for each parameter combination
                print(f"Method: {method}, Crop Size: {crop_size}, std: {std},Dataset: {DATASET.__name__}, Model: {MODEL.__name__}, Adaptive Center: {adaptive_center}")
                res = run_trial(MODEL, DATASET,method, crop_size, adaptive_center,min_max , 
                                        std, 1, num_workers, pretrain_epoch, batchsize,
                                        hidden_dim, clf_epochs) 
                                
                    
              
                test_accuracy, val_accuracy, train_accuracy = res
                results[method][crop_size][std].append((test_accuracy, val_accuracy, train_accuracy))
   
    # Save the final results to a JSON file
    final_results = {
        'results': results,
        'epoch': pretrain_epoch,
        'num_classes_workers': num_workers,
        'hidden_dim': hidden_dim,
        'batch_size': batchsize,
        'clf_epochs': clf_epochs,
        'Job_id': job_id,
        'method': methods,
        'crop_size': crop_sizes,
        'std': stds,
        'num_of_trials': num_of_trials,     
        'dataset': DATASET.__name__,
        'model': MODEL.__name__,  
        'adaptive_center': adaptive_center,
        'min_max': min_max,
    }
    print('Results:')
    print(final_results)
    if adaptive_center:
        job_id = job_id+'_adaptive_'
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_')
    with open(f'{job_id}_{method}_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f)
        
    print('Results saved successfully!')
    print('job_id:', job_id)
    print('Timestamp:')
    print(timestamp)
   
    