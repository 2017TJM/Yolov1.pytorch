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
        conf_loss = 0
        coord_loss = 0
        class_loss = 0

        output = output.sigmoid().view(batch_size*self.S*self.S, self.B*5 + self.C)
        target = self._encode(bboxes_gt, labels).view(batch_size*self.S*self.S, self.B*5 + self.C).to(device)

        # mask
        obj_mask = (target[:, 4] == 1)
        noobj_mask = (target[:, 4] == 0)

        # noobj loss
        # noobj conf loss
        noobj_pred = output[noobj_mask, :]
        noobj_pred_conf = noobj_pred[:, [5*i + 4 for i in range(self.B)]]
        noobj_target = target[noobj_mask, :]
        noobj_target_conf = noobj_target[:, [5*i + 4 for i in range(self.B)]]

        conf_loss = self.noobj_scale * t.sum((noobj_pred_conf - noobj_target_conf)**2)

        # obj loss
        obj_pred = output[obj_mask, :]
        obj_target = target[obj_mask, :]
        # match
        for i in range(obj_target.shape[0]):
            bbox_target = obj_target[i, :4].view(1, 4)
            bboxes_pred = []
            bboxes_conf = []
            for j in range(self.B):
                bboxes_pred.append(obj_pred[i, j*5:j*5 + 5][:4])
                bboxes_conf.append(obj_pred[i, j*5:j*5 + 5][-1])
            
            bboxes_pred = t.stack(bboxes_pred, dim=0)
            bboxes_conf = t.tensor(bboxes_conf)
            # xywh to xyxy
            bbox_target_xyxy = t.zeros_like(bbox_target).to(device)
            bbox_pred_xyxy = t.zeros_like(bboxes_pred).to(device)

            bbox_target_xyxy[:, :2] = bbox_target[:, :2] - bbox_target[:, 2:] / 2
            bbox_target_xyxy[:, 2:] = bbox_target[:, :2] + bbox_target[:, 2:] / 2

            bbox_pred_xyxy[:, :2] = bboxes_pred[:, :2] - bboxes_pred[:, 2:]**2 / 2
            bbox_pred_xyxy[:, 2:] = bboxes_pred[:, :2] + bboxes_pred[:, 2:]**2 / 2

            indices = match(bbox_target_xyxy, bbox_pred_xyxy)
            bbox_target[:, 2:] = t.sqrt(bbox_target[:, 2:])
            coord_loss = coord_loss + self.coord_scale * t.sum((bbox_target - bboxes_pred[indices, :])**2)
            conf_loss = conf_loss + t.sum((1 - bboxes_conf[indices])**2)

        # class_prob loss
        obj_pred_class = obj_pred[:, self.B*5:]
        obj_target_class = obj_target[:, self.B*5:]
        class_loss = t.sum((obj_pred_class - obj_target_class)**2)
        
        total_loss = coord_loss + conf_loss + class_loss

        return total_loss / batch_size, coord_loss / batch_size, conf_loss / batch_size, class_loss / batch_size

    def _encode(self, bboxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        batch_size = len(bboxes)
        target = t.zeros(batch_size, self.S, self.S, self.B*5 + self.C)
        # TODO
        # using config
        bboxes = [bbox.float() / 448 for bbox in bboxes]

        for b in range(batch_size):
            wh = bboxes[b][:,2:] - bboxes[b][:,:2]
            center_xy = (bboxes[b][:,2:] + bboxes[b][:,:2]) / 2

            for i in range(center_xy.shape[0]):
                center_xy_sample = center_xy[i]
                grid_coord = (center_xy_sample * self.S).ceil()-1
                
                xy = grid_coord.float() / self.S
                delta_xy = (center_xy_sample -xy) * self.S

                for j in range(self.B):
                    # Confidence of the bbox
                    target[b, int(grid_coord[1]), int(grid_coord[0]), j*5 + 4] = 1
                    # Coords
                    target[b, int(grid_coord[1]),int(grid_coord[0]), j*5:j*5 + 2] = delta_xy
                    target[b, int(grid_coord[1]),int(grid_coord[0]), j*5 + 2:j*5 + 4] = wh[i]
                    
                target[b, int(grid_coord[1]), int(grid_coord[0]), int(labels[b][i]) + self.B*5] = 1

        return target