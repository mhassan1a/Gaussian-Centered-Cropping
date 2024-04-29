from torchvision.models import resnet18, resnet34
import torch.nn as nn
import torch as torch


class Proto18(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Proto18, self).__init__()
        
        self.resnet = resnet18(
                                weights=None,
                               num_classes=hidden_dim,
                              )
        
        self.resnet.conv1 = nn.Conv2d(
                                      3, 64,
                                      kernel_size=3, stride=1,
                                      padding=1,
                                      bias=False
                                     )
        self.resnet.maxpool = nn.Identity()
    def forward(self, x):
        return self.resnet(x)


class Proto34(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Proto34, self).__init__()
        
        self.resnet = resnet34(
                                weights=None,
                               num_classes=hidden_dim,
                              )
        
        self.resnet.conv1 = nn.Conv2d(
                                      3, 64,
                                      kernel_size=3, stride=1,
                                      padding=1,
                                      bias=False
                                     )
        self.resnet.maxpool = nn.Identity()
    def forward(self, x):
        return self.resnet(x)
 
 
class Classifier(nn.Module):
    def __init__(self, num_features_in, hidden_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
                                nn.Linear(num_features_in, num_classes),  
                                )  
        
    def forward(self, x):
        return self.fc(x)
    
    

if __name__ == '__main__':      
    model= Proto18(hidden_dim=128)
    print(model)
    input = torch.randn(1, 32, 32, 3).permute(0, 3, 1, 2)
    output = model(input)   
    
    model= Proto34(hidden_dim=128)
    print(model)
    output = model(input)
    
    classifier = Classifier(128)
    print(classifier)