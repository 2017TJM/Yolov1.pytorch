from model.backbone import Backbone
from model.head import DetectionHead
from torch.nn import Module

class Yolo(Module):
    def __init__(self, head):
        super(Yolo, self).__init__()
        self.head = head
        self.backbone = Backbone()

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
