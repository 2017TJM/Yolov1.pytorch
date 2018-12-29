import torch as t
from torch.nn import Module
from util.bbox_util import match, xywh_to_xyxy, IOU, xyxy_to_xywh
import torch.nn.functional as F
import numpy as np

class YoloLoss(Module):
    def __init__(self, S=7, B=5, C=3, feat_stride=64, coord_scale=5.0, noobj_scale=0.5):
        super(YoloLoss, self).__init__()
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.S = S
        self.B = B
        self.C = C
        self.feat_stride = feat_stride

    def forward(self, output, bboxes_gt, labels):
        """
        Compute yolo loss

        Parameters
        ----------
        output: [batch_size, S*S*(C + B*5)]
        bboxes_gt: (list), [[num_bbox_gt, 4]]
        labels: (list), [[C,]]

        Returns
        -------

        """   
        device = output.device
        batch_size = output.shape[0]
        index1 = self.S*self.S*self.C
        index2 = index1 + self.S*self.S*self.B
        # [batch_size, S*S, C]
        class_prob = output[:, :index1].view(batch_size, self.S*self.S, self.C)
        # [batch_size, S*S, B]
        conf = output[:, index1:index2].view(batch_size, self.S*self.S, self.B)
        # [batch_size, S*S*B, 4]
        bboxes = output[:, index2:].view(batch_size, self.S*self.S*self.B, 4)

        conf_loss = 0
        coord_loss = 0
        class_loss = 0
        # iterate over batch
        for b, (bbox_gt, target) in enumerate(zip(bboxes_gt, labels)):
            center = (bbox_gt[:, :2] + bbox_gt[:, 2:]) / 2
            center /= self.feat_stride
            # [batch_size, 2] ------ (x, y)
            center = center.long()
            # [batch_size, ]
            obj_indices = center[:, 1]*center.shape[1] + center[:, 0]
            noobj_indices = t.tensor(list(set(range(self.S*self.S)) - set(obj_indices.cpu().numpy().tolist()))).long()

            # class loss
            class_loss += t.mean((class_prob[b, obj_indices, :] - target.view(-1, 1).expand(-1, self.C).float())**2)
            # part of noobj conf loss
            conf_loss += self.noobj_scale * t.mean(conf[b, obj_indices, :]**2)
            # match bbox with ground truth
            # [num_bbox, 4]
            bbox_pred = bboxes[b, obj_indices, :]
            bbox_gt_xywh = xyxy_to_xywh(bbox_gt, self.S, self.feat_stride)
            matched_indices = match(bbox_pred, bbox_gt_xywh)
            not_matched_indices = t.tensor(list(set(range(bbox_pred.shape[0])) - set(matched_indices.cpu().numpy().tolist()))).long()
            # obj coord los
            coord_loss += self.coord_scale * t.mean((bbox_pred[matched_indices] - bbox_gt_xywh)**2)
            # another part of noobj conf loss
            conf_loss += t.mean((1 - conf[b].view(-1)[not_matched_indices])**2)
        
        total_loss = coord_loss + conf_loss + class_loss

        return total_loss