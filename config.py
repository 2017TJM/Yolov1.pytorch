
class Config(object):
    is_cuda = True
    # num_classes
    C = 20
    # num_bboxes
    B = 2
    S = 7
    downsample_ratio=64
    input_size = (448, 448)
    detection_head_input_size = (7, 7)
    data_dir = r"F:\Code\Yolov1\dataset"
    feat_stride = 64
    shape_bbox_colors = ((179, 238, 58),(205, 0, 0), (205, 105, 201))
    batch_size = 2

cfg = Config()