from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
import numpy as np
from PIL import Image
from ImageNetLoad import ImageNetDownSample
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class TwoViewCifar10(CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, cropping=None):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.cropping = cropping
    def __getitem__(self, index):
        x, target = self.data[index], self.targets[index]

        if self.cropping is not None:
            view1,view2 = self.cropping.gcc(x)
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

        if self.cropping is not None:
            view1,view2 = self.cropping.gcc(x)
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

        if self.cropping is not None:
            view1, view2 = self.cropping.gcc(x)
        else:
            view1, view2 = x, x
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return view1, view2, target
    

if __name__ == '__main__':
    train_dataset = TwoViewCifar10(root='./data', train=True, download=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break
    
    train_dataset = TwoViewImagenet64(root='./data/Imagenet64_train', train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break
    
    train_dataset = TwoViewCifar100(root='./data', train=True, download=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for x1, x2, target in train_loader:
        print(x1.shape, x2.shape, target.shape)
        break