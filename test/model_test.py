import sys
sys.path.append('../')
import unittest
import HTMLTestRunner
import torch as t
from config import cfg
from model.backbone import Backbone
from model.head import ClassificationHead, DetectionHead
from model.yolo import Yolo

device = t.device("cuda:0" if cfg.is_cuda else "cpu")

class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.backbone = Backbone().to(device)
        self.yolo_c_head = Yolo(head=ClassificationHead()).to(device)
        self.yolo_d_head = Yolo(DetectionHead(input_size=cfg.detection_head_input_size, C=cfg.C, B=cfg.B)).to(device)
        self.x = t.ones(1, 3, 448, 448).to(device)

    def test_backbone(self):
        self.assertEqual(self.backbone(self.x).shape, (1, 1024, 14, 14))

    def test_classification_head(self):
        self.assertEqual(self.yolo_c_head(self.x).shape, (1, 1000))

    def test_detection_head(self):
        self.assertEqual(self.yolo_d_head(self.x).shape, (1, 49, 30))

if __name__ == "__main__":
    unittest.main()