import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# Build Resnet 18 to cifar10


class BasicBlock(nn.Module):
    def __init__(self,in_features, out_features,stride,kernel_size=3):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_features,out_features,kernel_size,stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features,out_features,kernel_size,stride=1, padding=1)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features,out_features,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self,x):
        out = F.relu(self.bn(self.conv1(x)))
        out = F.relu(self.bn(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self,block,num_classes):
        super(Resnet,self).__init__()
        
        self.conv1 = self.conv_bn(3,64,3,1)

        self.layer1 = self._make_layer(block,64,64,kernel_size=3,stride=1)
        self.layer2 = self._make_layer(block,64,128,kernel_size=3,stride=2)
        self.layer3 = self._make_layer(block,128,256,kernel_size=3,stride=2)
                
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = nn.Linear(4*4*256,num_classes) # and softmax

    def _make_layer(self,block,in_features,out_features,kernel_size=3,stride=1):
        layer = nn.Sequential(
            block(in_features, out_features,stride=stride),
            block(out_features,out_features,stride=1)
        )
        return layer

    def conv_bn(self,in_features,out_features,kernel_size=3,stride=1):
        return nn.Sequential( nn.Conv2d(in_features,out_features,kernel_size,stride,padding=1),
                    nn.BatchNorm2d(out_features) )


    def forward(self,x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out


def Resnet20(num_classes):
    return Resnet(BasicBlock, num_classes)
