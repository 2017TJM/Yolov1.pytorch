
class Config(object):
    is_cuda = True
    # num_classes
    C = 20
    # num_bboxes
    B = 2
    downsample_ratio=64
    detection_head_input_size = (7, 7)

cfg = Config()