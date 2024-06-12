import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class QuantModel(nn.Module):
    def __init__(self,input_model):
        super(QuantModel, self).__init__()
        self.quant = QuantStub()
        self.model=input_model
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
        
    def fuse_model(self):
        for m in self.modules():
            if type(m) == MyModel:
                print(type(m))
                torch.quantization.fuse_modules(m, [['conv1', 'relu'], ], inplace=True)
                print("Fused")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.dequant(x)
        return x
