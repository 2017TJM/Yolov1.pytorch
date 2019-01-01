import torch as t

def intersect(bboxes_a, bboxes_b):
    # [num_bbox_a, num_bbox_b, 4]
    A = bboxes_a.shape[0]
    B = bboxes_b.shape[0]

    bboxes_a = bboxes_a.view(-1, 1, 4)
    bboxes_b = bboxes_b.view(1, -1, 4)

    bboxes_a = bboxes_a.expand(-1, B, -1)
    bboxes_b = bboxes_b.expand(A, -1, -1)

    xy_min = t.max(bboxes_a[:, :, :2], bboxes_b[:, :, :2])
    xy_max = t.min(bboxes_a[:, :, 2:], bboxes_b[:, :, 2:])

    wh = t.clamp((xy_max - xy_min), min=0, max=10000)
    return wh[:, :, 0] * wh[:, :, 1]

def IOU(bboxes_a, bboxes_b):
    A = bboxes_a.shape[0]
    B = bboxes_b.shape[0]

    bbox_a_wh = bboxes_a[:, 2:] - bboxes_a[:, :2]
    bbox_a_area = bbox_a_wh[:, 0] * bbox_a_wh[:, 1]
    bbox_a_area = bbox_a_area.view(-1, 1)
    bbox_a_area = bbox_a_area.expand(-1, B)

    bbox_b_wh = bboxes_b[:, 2:] - bboxes_b[:, :2]
    bbox_b_area = bbox_b_wh[:, 0] * bbox_b_wh[:, 1]
    bbox_b_area = bbox_b_area.view(1, -1)
    bbox_b_area = bbox_b_area.expand(A, -1)

    inter = intersect(bboxes_a, bboxes_b)
    union = bbox_a_area + bbox_b_area - inter + 1e-6
    ious = inter.float() / union.float()
    return t.clamp(ious, min=0, max=1)

def match(bbox_gt, bbox_pred):
    """

    Parameters
    ----------
    bbox_pred: [num_bbox_pred, 4]
    bbox_gt: [num_bbox_gt, 4]

    Returns
    -------
    indices: [num_bbox_gt, ]. The bboxes of bbox_pred which have the maximun IOU with bbox_gt
    """
    # [num_gt_bbox, num_bbox_pred]
    iou = IOU(bbox_gt, bbox_pred)
    # return t.argmax(iou, dim=1)
    return t.argmax(iou, dim=1)



def xywh_to_xyxy(bboxes, S=7, feat_stride=64):
    """
    Convert bbox format from (x, y, \sqrt{w}, \sqrt{h}) to (x_min, y_min, x_max, y_max).
    x and y is relative to the grid cell. w and h is relative to the whole image.
    
    Parameters
    ----------
    bboxes: [S*S, num_bbox, 4]
    S=7: The original image is divided into a grid of S*S
    feat_stride=64: Downsample ratio of the network

    Returns
    -------
    new_bboxes: Size [num_bbox, 4]. Format (x_min, y_min, x_max, y_max).

    """
    num_bbox = bboxes.shape[1]

    x_offset = t.arange(0, S).view(S, 1, 1) * feat_stride
    y_offset = x_offset.permute(1, 0, 2)

    bboxes = bboxes.view(S, S, num_bbox, 4)
    abs_bboxes = t.zeros_like(bboxes).view(S, S, num_bbox, 4)
    abs_bboxes[:, :, :, 0] = bboxes[:, :, :, 0] * feat_stride + x_offset.float()
    abs_bboxes[:, :, :, 1] = bboxes[:, :, :, 1] * feat_stride + y_offset.float()
    abs_bboxes[:, :, :, 2:] = bboxes[:, :, :, 2:]**2 * S * feat_stride 
    abs_bboxes = abs_bboxes.view(-1, num_bbox, 4)

    new_bboxes = t.zeros_like(abs_bboxes)
    w = abs_bboxes[:, :, 2]
    h = abs_bboxes[:, :, 3]
    new_bboxes[:, :, 0] = abs_bboxes[:, :, 0] - w / 2
    new_bboxes[:, :, 1] = abs_bboxes[:, :, 1] - h / 2
    new_bboxes[:, :, 2] = abs_bboxes[:, :, 0] + w / 2
    new_bboxes[:, :, 3] = abs_bboxes[:, :, 1] + h / 2

    new_bboxes = t.clamp(new_bboxes, min=0, max=S*feat_stride)
    
    return new_bboxes

def xyxy_to_xywh(bboxes, S=7, feat_stride=64):
    """
    Convert bbox format from (x_min, y_min, x_max, y_max) to (x, y, \sqrt{w}, \sqrt{h}).
    x and y is relative to the grid cell. w and h is relative to the whole image.
    
    Parameters
    ----------
    bboxes: [num_bbox, 4]
    S=7: The original image is divided into a grid of S*S
    feat_stride=64: Downsample ratio of the network

    Returns
    -------
    new_bboxes: Size [num_bbox, 4]. Format (x, y, \sqrt{w}, \sqrt{w}).
    """
    new_bboxes = t.zeros_like(bboxes, dtype=t.float32)

    wh = bboxes[:, 2:] - bboxes[:, :2]
    wh = t.sqrt(wh.float() / (feat_stride * S))

    center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    # Remove x_offset and y_offset
    center = center % feat_stride
    center = center.float() / feat_stride

    new_bboxes[:, :2] = center
    new_bboxes[:, 2:] = wh
    
    return new_bboxes