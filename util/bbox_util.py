import numpy as np

def intersect(bbox, bboxes):
    xy_min = np.maximum(bbox[:2], bboxes[:, :2])
    xy_max = np.minimum(bbox[2:], bboxes[:, 2:])

    wh = np.clip(xy_max - xy_min, a_min=0, a_max = np.inf)
    return wh[:, 0] * wh[:, 1]

def IOU(bbox, bboxes):
    bbox_wh = bbox[:2] - bbox[2:]
    bbox_area = bbox_wh[0] * bbox[1]

    bboxes_wh = bboxes[:, :2] - bboxes[:, :2]
    bboxes_area = bboxes_wh[:, 0] * bboxes_wh[:, 1]

    inter = intersect(bbox, bboxes)
    union = bbox_area + bboxes_area - inter
    
    return inter / union

    