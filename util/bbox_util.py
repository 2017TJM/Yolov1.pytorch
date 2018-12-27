import numpy as np

def intersect(bboxes_a, bboxes_b):
    """
    Compute the intersections
        :param bboxes_a: [num_bbox_a, 4]
        :param bboxes_b: [num_bbox_b, 4]
    """
    # [num_bbox_a, num_bbox_b, 4]
    A = bboxes_a.shape[0]
    B = bboxes_b.shape[0]

    bboxes_a = bboxes_a.reshape(-1, 1, 4)
    bboxes_b = bboxes_b.reshape(1, -1, 4)

    bboxes_a = np.tile(bboxes_a, (1, B, 1))
    bboxes_b = np.tile(bboxes_b, (A, 1, 1))

    xy_min = np.maximum(bboxes_a[:, :, :2], bboxes_b[:, :, :2])
    xy_max = np.minimum(bboxes_a[:, :, 2:], bboxes_b[:, :, 2:])

    wh = np.clip((xy_max - xy_min), a_min=0, a_max = np.inf)
    return wh[:, :, 0] * wh[:, :, 1]

def IOU(bboxes_a, bboxes_b):
    A = bboxes_a.shape[0]
    B = bboxes_b.shape[0]

    bbox_a_wh = bboxes_a[:, 2:] - bboxes_a[:, :2]
    bbox_a_area = bbox_a_wh[:, 0] * bbox_a_wh[:, 1]
    bbox_a_area = bbox_a_area.reshape(-1, 1)
    bbox_a_area = np.tile(bbox_a_area, (1, B))

    bbox_b_wh = bboxes_b[:, 2:] - bboxes_b[:, :2]
    bbox_b_area = bbox_b_wh[:, 0] * bbox_b_wh[:, 1]
    bbox_b_area = bbox_b_area.reshape(1, -1)
    bbox_b_area = np.tile(bbox_b_area, (A, 1))

    
    inter = intersect(bboxes_a, bboxes_b)
    union = bbox_a_area + bbox_b_area - inter
    
    return inter / union

def match(bbox_pred, bbox_gt):
    """
    Match bbox from prediction with ground-truth bbox
        :param bbox_pred: [num_cell, num_bbox, 5]
        :param bbox_gt: [num_gt_bbox, 4]
    """
    bbox_pred = bbox_pred.reshape(-1, 5)
    # [num_gt_bbox, num_cell*num_bbox]
    iou = IOU(bbox_gt, bbox_pred[:, 1:])
    return np.argmax(iou, axis=1)

def xywh_to_xyxy(bboxes, S=7, feat_stride=64):
    """
    Convert bbox predictions of yolo from (x, y, w, h) to (x_min, y_min, x_max, y_max)
        :param bboxes: [num_cell, num_bbox, 5]
        :param feat_stride=64: 
    """
    num_bbox = bboxes.shape[1]

    x_offset = np.arange(0, S).reshape(S, 1, 1) * feat_stride
    y_offset = np.transpose(x_offset, [1, 0, 2])

    bboxes = bboxes.reshape(S, S, num_bbox, 5)
    bboxes[:, :, :, 1] = bboxes[:, :, :, 1] * feat_stride + x_offset
    bboxes[:, :, :, 2] = bboxes[:, :, :, 2] * feat_stride + y_offset
    bboxes[:, :, :, 3:] = bboxes[:, :, :, 3:] * S * feat_stride 
    bboxes = bboxes.reshape(-1, num_bbox, 5)

    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, :, 0] = bboxes[:, :, 0]
    new_bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] / 2
    new_bboxes[:, :, 2] = bboxes[:, :, 2] - bboxes[:, :, 4] / 2
    new_bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3] / 2
    new_bboxes[:, :, 4] = bboxes[:, :, 2] + bboxes[:, :, 4] / 2

    new_bboxes = np.clip(new_bboxes, a_min=0, a_max=S*feat_stride)
    return new_bboxes
    
