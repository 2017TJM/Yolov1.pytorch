import json
import os
from PIL import Image
import torch as t

class ShapeDataset(object):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.imgdir = os.path.join(root, 'images')
        annfile = os.path.join(root, split + '.json')
        self.ann = json.load(open(annfile, 'r'))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        annotation = self.ann[index]
        pathname = annotation["image"]["pathname"]
        objects = annotation["objects"]
        bboxs = []
        labels = []

        for bbox in objects:
            # top left
            x_min = bbox["bounding_box"]["minimum"]["c"]
            y_min = bbox["bounding_box"]["minimum"]["r"]
            # bottom right
            x_max = bbox["bounding_box"]["maximum"]["c"]
            y_max = bbox["bounding_box"]["maximum"]["r"]
            bboxs.append([x_min, y_min, x_max, y_max])
            labels.append(LABEL_TO_NUMBER[bbox["category"]])
        
        img = Image.open(pathname)
        if self.transform:
            img = self.transform(img)
        bboxs = t.Tensor(bboxs)
        labels = t.Tensor(labels).int()

        return img, bboxs, labels

LABEL_TO_NUMBER = {
    "circle": 0,
    "triangle": 1,
    "rectangle": 2
}
