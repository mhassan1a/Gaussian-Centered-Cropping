from misc import EarlyStopper
from datasets import TwoViewDataSet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from cropping import GaussianCrops
from models import Proto18
import numpy as np
import torch
import torch.nn as nn
from lightly.loss import NegativeCosineSimilarity, ntx_ent_loss
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms
import json
from datetime import datetime

def train(epoch,max_epochs, model, dataloader, optimizer, scheduler, criterion, device):
    #one epoch of training
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

def pretraining(max_epochs=100, batch_size=512, num_workers=40, cropping=None, transform=None, hidden_dim=256):
    print(f"GPU Availble:{torch.cuda.is_available()}")
    print('Pre-Traing Starts ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_dataset = TwoViewDataSet(root='./data', train=True, download=True, cropping=cropping, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = Proto18(hidden_dim = hidden_dim)
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,60,80,100,150], gamma=0.1)
    criterion = ntx_ent_loss.NTXentLoss(temperature = 0.1, memory_bank_size=0)
    train_progress = tqdm(range(max_epochs), desc='pre training', unit='epoch')
    pretrain_loss = []
    for epoch in train_progress:
        loss = train(epoch, max_epochs, model, train_loader, optimizer, scheduler, criterion, device)
        pretrain_loss.append(loss)
        train_progress.set_postfix(loss=loss)
    return model, pretrain_loss

def addclassifier(model, num_classes):
    for name , param in model.resnet.named_parameters():
        param.requires_grad = False

    num_features = model.resnet.fc.in_features
    classifier = nn.Sequential(
    nn.Linear(num_features, 1024),  
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),  
    nn.Linear(1024, 10),
    )  

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
            ),
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers =num_workers )
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers =num_workers )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers =num_workers )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9
            )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
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
            #print("We are at epoch:", epoch)
            break
        
        #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # After training, evaluate the model on the test set
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
        #print(f"Test Accuracy: {test_accuracy:.4f}")
        
    return test_accuracy , val_accuracy, train_accuracy


if __name__ == '__main__':
    # config
    pretrain_epoch = 200
    num_workers = 4
    hidden_dim = 256
    batchsize = 512
    clf_epochs = 150


    methods = ['GCC_NO_regularization','GCC_regularization']
    crop_sizes = [0.4, 0.6, 0.8]
    num_of_trials = 5
    stds = [0.001, 0.01, 0.1, 0.3,0.5,0.7, 1,2,5,10,100,200]
    results = {}
    for method in methods:
        results[method] = {}
        for crop_size in crop_sizes:
            results[method][crop_size] = {}
            for std in stds:
                results[method][crop_size][std] = []
                for trial in range(num_of_trials):
                    view_transform = transforms.Compose(
                        [   transforms.ToPILImage(),
                            #transforms.RandomCrop(24),
                            transforms.RandomHorizontalFlip(),
                            #transforms.ColorJitter(brightness=.5, hue=.3),
                            transforms.GaussianBlur(kernel_size=(3,3)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean = (0.4914, 0.4822, 0.4465),
                                std = (0.2023, 0.1994, 0.2010)
                            ),
                        ]
                    )  
                    crop_percentage = crop_size
                    seed = None
                    std_scale = std
                    pad = True 
                    reg = False
                    if method == 'GCC_regularization':
                        pad = False 
                        reg = True
                
                
                    crop = GaussianCrops(crop_percentage = crop_percentage,
                                          seed = seed, 
                                        std_scale = std_scale,
                                       padding=pad,regularised_crop=reg)
                    
                    model, pretrain_loss = pretraining(max_epochs= pretrain_epoch,
                                                        batch_size=batchsize,
                                                          num_workers= num_workers,
                                                            cropping=crop, 
                                                            transform=view_transform,
                                                            hidden_dim=hidden_dim)

                    model = addclassifier(model, 10)
                    test_accuracy , val_accuracy, train_accuracy = train_classifier(model=model, 
                                                                                    max_epochs=clf_epochs,
                                                                                    earlystop_patience=10,
                                                                                    num_workers= num_workers,
                                                                                    )

                    results[method][crop_size][std].append((test_accuracy, val_accuracy, train_accuracy))
                    print(f"Method: {method}, Crop Size: {crop_size}, std: {std}, Trial: {trial}, Test Accuracy: {test_accuracy}, Val Accuracy: {val_accuracy}, Train Accuracy: {train_accuracy}")
                    
    print(results)
    results['epoch'] = pretrain_epoch
    results['num_classes_workers'] = num_workers
    results['hidden_dim'] = hidden_dim
    results['batch_size'] = batchsize

    timestamp = datetime.now()
    with open(f'results_{timestamp}_.json', 'w') as f:
        json.dump(results, f)
