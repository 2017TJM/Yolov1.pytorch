import cv2
import numpy as np
import torch as t
import sys
import random
import math
sys.path.append('../')
from util.bbox_util import intersect

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels

class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img, bboxes=None, labels=None):
        return self.lambd(img, bboxes, labels)

class ConvertFromInts(object):
    def __call__(self, img, bboxes=None, labels=None):
        return img.astype(np.float32), bboxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, bboxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), bboxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img, bboxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, bboxes, labels

class ToCV2Image(object):
    def __call__(self, tensor, bboxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), bboxes, labels

class ToTensorImage(object):
    def __call__(self, cvimage, bboxes=None, labels=None):
        return t.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), bboxes, labels


class Crop(object):

    def __init__(self, max_crop_ratio=0.2, size=None):
        self.max_crop_ratio = max_crop_ratio
        self.size = size

    def __call__(self, img, bboxes=None, labels=None):

        if self.size != None:
            rect = self._fixed_size_rect(img)
        else:
            rect = self._max_ratio_rect(img)
        # Crop the input image
        img = img[rect[1]:rect[3], rect[0]:rect[2], :]
        
        # remove bbox beyond the croped image
        intersections = intersect(rect, bboxes)
        bboxes = bboxes[intersections > 0]

        bboxes[:, :2] = np.maximum(bboxes[:, :2], rect[:2])
        bboxes[:, :2] -= rect[:2]
        bboxes[:, 2:] = np.minimum(bboxes[:, 2:], rect[2:])
        bboxes[:, 2:] -= rect[:2]
        
        return img, bboxes, labels

    def _fixed_size_rect(self, img):
        height, width = img.shape[:2]
        w = self.size
        h = self.size

        x_min = random.randrange(width - w)
        y_min = random.randrange(height - h)
        x_max = x_min + w
        y_max = y_min + h

        return np.array([x_min, y_min, x_max, y_max])

    def _max_ratio_rect(self, img):
        height, width = img.shape[:2]

        crop_ratio = np.random.uniform(0, self.max_crop_ratio, size=4)

        x_min = int(crop_ratio[0] * width)
        y_min = int(crop_ratio[1] * height)
        x_max = int(width - crop_ratio[2] * width - 1)
        y_max = int(height - crop_ratio[3] * height - 1)
        
        return np.array([x_min, y_min, x_max, y_max])

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 1):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, img, bboxes=None, labels=None):
        in_size = img.shape
        img = cv2.resize(img, (self.size, self.size))
        out_size = img.shape

        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        bboxes[:, 0] = x_scale * bboxes[:, 0]
        bboxes[:, 2] = x_scale * bboxes[:, 2]
        bboxes[:, 1] = y_scale * bboxes[:, 1]
        bboxes[:, 3] = y_scale * bboxes[:, 3]
        bboxes = bboxes.astype(np.int32)

        return img, bboxes, labels

class YoloAugmentation(object):
    def __init__(self, size=448, mean=(125, 125, 125)):
        self.mean = mean
        self.size = size
        self.pipe = Compose([
            ConvertFromInts(),
            ConvertColor(),
            RandomBrightness(),
            RandomSaturation(),
            ConvertColor(current="HSV", transform="BGR"),
            Resize(size),
            SubtractMeans(mean)
        ])

    def __call__(self, img, bboxes, labels):
        return self.pipe(img, bboxes, labels)

