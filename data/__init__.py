import torch as t

def detection_collate(batch):
    imgs = []
    bboxes = []
    labels = []

    for sample in batch:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])

    return t.stack(imgs, 0), bboxes, labels