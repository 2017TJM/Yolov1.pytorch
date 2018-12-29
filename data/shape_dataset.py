import json
import os
import cv2
import torch as t
import numpy as np

class ShapeDataset(object):
    def __init__(self, root, split='train', transform=None):
        self.imgdir = os.path.join(root, 'images')
        annfile = os.path.join(root, split + '.json')
        self.ann = json.load(open(annfile, 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        annotation = self.ann[index]
        pathname = annotation["image"]["pathname"]
        objects = annotation["objects"]
        bboxes = []
        labels = []

        for bbox in objects:
            # top left
            x_min = bbox["bounding_box"]["minimum"]["c"]
            y_min = bbox["bounding_box"]["minimum"]["r"]
            # bottom right
            x_max = bbox["bounding_box"]["maximum"]["c"]
            y_max = bbox["bounding_box"]["maximum"]["r"]
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(LABEL_TO_NUMBER[bbox["category"]])
        
        img = cv2.imread(pathname)
        bboxes = np.array(bboxes)
        labels = np.array(labels)

        if self.transform:
            img, bboxes, labels = self.transform(img, bboxes, labels)
        img = t.from_numpy(img).permute(2, 0, 1)
        bboxes = t.from_numpy(bboxes).int()
        labels = t.from_numpy(labels).int()

        return img, bboxes, labels

LABEL_TO_NUMBER = {
    "circle": 0,
    "triangle": 1,
    "rectangle": 2
}
