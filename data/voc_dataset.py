import os
import cv2
import torch as t
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VocDataset(Dataset):
    def __init__(self, root, year="2007", split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.__imgpath = os.path.join(root, 'VOC' + year, "JPEGImages", "{id}.jpg")
        self.__annpath = os.path.join(root, 'VOC' + year, "Annotations", "{id}.xml")
        self.ids = []

        with open(os.path.join(root, 'VOC' + year, 'imageSets\Main', split + '.txt'), 'r') as fin:
            for line in fin.readlines():
                self.ids.append(line.strip())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        imgpath = self.__imgpath.format(id=id)
        annpath = self.__annpath.format(id=id)

        # TODO
        # determine using difficult bbox or not
        bboxes, labels, difficult = self._parse(annpath)
        img = cv2.imread(imgpath)
        if self.transform:
            img, bboxes, labels = self.transform(img, bboxes, labels)

        return t.from_numpy(img).permute(2, 1, 0), t.from_numpy(bboxes), t.from_numpy(labels)

    def _parse(self, annpath):
        tree = ET.parse(annpath)
        root = tree.getroot()
        bboxes = []
        labels = []
        difficult = []

        for obj in root.iter('object'):
            name = obj.find('name').text
            is_difficult = int(obj.find('difficult').text)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            

            labels.append(VOC_CLASSES.index(name))
            bboxes.append([xmin, ymin, xmax, ymax])
            difficult.append(is_difficult)
        
        return np.array(bboxes), np.array(labels), np.array(difficult)


