import torch as t
import torch.nn.functional as F
from torch.nn import Module, Sequential, LeakyReLU, Linear, Conv2d, Dropout

class ClassificationHead(Module):
    
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.classifier = Linear(in_features=1024, out_features=1000)
            
    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, 1024)
        return self.classifier(x)

class DetectionHead(Module):
    def __init__(self, input_size=(7, 7), C=20, B=2):
        super(DetectionHead, self).__init__()
        self.input_size = input_size
        self.B = B
        self.C = C
        self.feature = Sequential(
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=2, padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            LeakyReLU(0.1),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            LeakyReLU(0.1)
        )
        self.regresor = Sequential(
            Linear(in_features=1024*input_size[0]*input_size[1], out_features=4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(in_features=4096, out_features=input_size[0]*input_size[1]*(C + B*5))
        )
    
    def forward(self, x):
        x = self.feature(x).view(-1, )
        x = self.regresor(x)
        # return x.view(-1, self.input_size[0]*self.input_size[1], self.B*5 + self.C)
        return x