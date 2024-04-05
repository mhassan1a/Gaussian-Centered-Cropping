from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, CocoDetection
import numpy as np
from PIL import Image
from ImageNetLoad import ImageNetDownSample
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms as transforms
from cropping import GaussianCrops
import torch
import os


class TwoViewCifar10(CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, cropping=None):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.cropping = cropping
    def __getitem__(self, index):
        x, target = self.data[index], self.targets[index]
         
        if isinstance(self.cropping, GaussianCrops):
            view1,view2 = self.cropping.gcc(x)

        elif isinstance(self.cropping, RandomCrop):
            x = torch.from_numpy(x).permute(2,0,1)
            view1 = self.cropping(x).permute(1,2,0).numpy() 
            view2 = self.cropping(x).permute(1,2,0).numpy()
        else:
            view1,view2 = x, x
            
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
            
        return view1, view2, target


class TwoViewCifar100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, cropping=None):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.cropping = cropping
    def __getitem__(self, index):
        x, target = self.data[index], self.targets[index]

        if isinstance(self.cropping, GaussianCrops):
            view1,view2 = self.cropping.gcc(x)

        elif isinstance(self.cropping, RandomCrop):
            x = torch.from_numpy(x).permute(2,0,1)
            view1 = self.cropping(x).permute(1,2,0).numpy() 
            view2 = self.cropping(x).permute(1,2,0).numpy()
        else:
            view1,view2 = x, x
            
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
            
        return view1, view2, target
    
    
class TwoViewImagenet64(ImageNetDownSample):
    def __init__(self, root=None, train=True, transform=None, target_transform=None,
                 download=False, cropping=None):
        
        super().__init__(root=root, train=train,
                         transform=transform, target_transform=target_transform)
        self.cropping = cropping
          
    def __getitem__(self, index):
        x, target = self.train_data[index], self.train_labels[index]
        if isinstance(self.cropping, GaussianCrops):
            view1,view2 = self.cropping.gcc(x)

        elif isinstance(self.cropping, RandomCrop):
            x = torch.from_numpy(x).permute(2,0,1)
            view1 = self.cropping(x).permute(1,2,0).numpy() 
            view2 = self.cropping(x).permute(1,2,0).numpy()
        else:
            view1,view2 = x, x
            
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
            
        return view1, view2, target
    
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, 
                 cropping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        self.cropping = cropping
        self.transform = transform

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name, 'images')
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, target = self.images[index]
        x = Image.open(img_path).convert('RGB')
        x= np.array(x)
        if isinstance(self.cropping, GaussianCrops):
            view1,view2 = self.cropping.gcc(x)

        elif isinstance(self.cropping, RandomCrop):
            x = torch.from_numpy(x).permute(2,0,1)
            view1 = self.cropping(x).permute(1,2,0).numpy() 
            view2 = self.cropping(x).permute(1,2,0).numpy()
        else:
            view1,view2 = x, x
            
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
            
        return view1, view2, target
    

if __name__ == '__main__':
    
    view_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(8, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    
    train_dataset = TwoViewCifar10(root='./data', train=True, download=False, transform=view_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break
    
    
    import time
    start = time.time()
    train_dataset = TwoViewImagenet64(root='./data/Imagenet64_train', train=True, transform=view_transform)
    train_loader = DataLoader(train_dataset, batch_size=2,num_workers=0, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break

    
    train_dataset = TwoViewCifar100(root='./data', train=True, download=False, transform=view_transform)
    train_loader = DataLoader(train_dataset, batch_size=3,num_workers=0, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break

    cropping = GaussianCrops(0.2,1)    
    train_dataset = TwoViewCifar10(root='./data', train=True, cropping=cropping, transform=view_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break
    
    cropping = RandomCrop(24)
    train_dataset = TwoViewCifar10(root='./data', train=True, download=False,transform=view_transform, cropping=cropping)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break