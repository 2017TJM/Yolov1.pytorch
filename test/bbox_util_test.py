import sys
sys.path.append('../')
import unittest
import torch as t
from config import cfg
from util.bbox_util import intersect, IOU, match, xyxy_to_xywh, xywh_to_xyxy

class BboxUtilTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.feat_stride = 16
        self.S = 2
        self.normal_bbox = t.tensor([[20, 20, 150, 150], [30, 30, 160, 160], [80, 80, 300, 300], [95, 95, 350, 350]])
        self.xywh_bbox = t.tensor([[0.1, 0.5, 0.6, 0.4], [0.2, 0.3, 0.7, 0.4], [0.5, 0.7, 0.3, 0.1], [0.8, 0.1, 0.3, 0.5]]).view(self.S*self.S, -1, 4)
        self.xyxy_bbox = t.tensor([[ 0,  5.44,  7.36, 10.56], [ 0, 18.24, 11.04, 23.36], [22.56, 11.04, 25.44, 11.36], [27.36, 13.6, 30.24, 21.6]])
        
    def test_intersect(self):
        bbox_a = self.normal_bbox[:2, :]
        bbox_b = self.normal_bbox[2:, :]
        intersections = intersect(bbox_a, bbox_b)
        self.assertEqual(intersections[0, 0], 4900)
        self.assertEqual(intersections[0, 1], 3025)
        self.assertEqual(intersections[1, 0], 6400)
        self.assertEqual(intersections[1, 1], 4225)

    def test_IOU(self):
        bbox_a = self.normal_bbox[:2, :]
        bbox_b = self.normal_bbox[2:, :]
        ious = IOU(bbox_a, bbox_b)
        self.assertAlmostEqual(ious[0, 0].item(), 0.0811, places=4)
        self.assertAlmostEqual(ious[0, 1].item(), 0.0383, places=4)
        self.assertAlmostEqual(ious[1, 0].item(), 0.1087, places=4)
        self.assertAlmostEqual(ious[1, 1].item(), 0.0544, places=4)
    
    def test_match(self):
        bbox_a = self.normal_bbox[:2, :]
        bbox_b = self.normal_bbox[2:, :]
        indices = match(bbox_a, bbox_b)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[1].item(), 0)

    def test_xyxy_to_xywh(self):
        xywh_bbox = xyxy_to_xywh(self.xyxy_bbox, S=self.S, feat_stride=self.feat_stride)
        self.assertAlmostEqual(xywh_bbox[3, 0].item(), 0.8000, places=4)
        self.assertAlmostEqual(xywh_bbox[3, 1].item(), 0.1000, places=4)
        self.assertAlmostEqual(xywh_bbox[3, 2].item(), 0.3000, places=4)
        self.assertAlmostEqual(xywh_bbox[3, 3].item(), 0.5000, places=4)

    def test_xywh_to_xyxy(self):
        xyxy_bbox = xywh_to_xyxy(self.xywh_bbox, S=self.S, feat_stride=self.feat_stride)
        self.assertAlmostEqual(xyxy_bbox[0, 0, 0].item(), 0.0000, places=4)
        self.assertAlmostEqual(xyxy_bbox[0, 0, 1].item(), 5.4400, places=4)
        self.assertAlmostEqual(xyxy_bbox[0, 0, 2].item(), 7.3600, places=4)
        self.assertAlmostEqual(xyxy_bbox[0, 0, 3].item(), 10.5600, places=4)


if __name__ == "__main__":
    unittest.main()