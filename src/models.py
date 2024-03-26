from torchvision.models import resnet18
import torch.nn as nn


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
                                      padding=0,
                                      bias=False
                                     )
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, ceil_mode=False)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = SimCLRProjectionHead(num_features, num_features*4, 10)
    def forward(self, x):
        return self.resnet(x)
    

if __name__ == '__main__':      
    model= Proto18(hidden_dim=128)
    print(model)