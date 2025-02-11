import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from collections import OrderedDict
class LeNet_TS(nn.Module):
    def __init__(self):
        super(LeNet_TS, self).__init__()

        self.layer1 = nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.layer2 = nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.layer3  =nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1)
        self.act= nn.ReLU()
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        out = self.layer1(x)
        out=self.act(out)
        out = self.layer2(out)
        out=self.act(out)
        out = self.layer3(out)
        out=self.act(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.layer1 = nn.Conv2d(1, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.layer2 = nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.layer3  =nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1)
        self.act= nn.ReLU()
        self.fc = nn.Linear(588, 10)

    def forward(self, x):
        out = self.layer1(x)
        out=self.act(out)
        out = self.layer2(out)
        out=self.act(out)
        out = self.layer3(out)
        out=self.act(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out