from model.head import DetectionHead
from model.yolo import Yolo
from data.shape_dataset import ShapeDataset
from data.preprocess import YoloAugmentation
from data import detection_collate
from util.vis_util import plot_bbox, plot_grid, plot_image
from metric.loss import YoloLoss
from config import cfg
import os
import torch as t
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import SGD


if __name__ == "__main__":
    device = t.device("cpu")
    root = os.path.join(cfg.data_dir, 'shape')
    
    transform = YoloAugmentation(size=448)
    dataset = ShapeDataset(root=root, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=detection_collate)

    # C = 3, B = 5 for shape dataset
    model = Yolo(DetectionHead(C=cfg.C, B=cfg.B)).to(device)
    trainer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criteria = YoloLoss(cfg.S, cfg.B, cfg.C, cfg.feat_stride)
    epoch = 1

    model.train()

    for e in range(epoch):
        train_loss = 0
        i = 0
        for imgs, bboxes, labels in loader:
            imgs = imgs.to(device)
            bboxes = [item.to(device) for item in bboxes]
            labels = [item.to(device) for item in labels]
            i += 1
            trainer.zero_grad()
            output = model(imgs)
            loss = criteria(output, bboxes, labels)
            loss.backward()
            trainer.step()
            train_loss += loss.cpu().item()
            print('Loss:', train_loss / i)
        print('Epoch %d  loss: %.3f' % (e, train_loss / i))
        t.save(model.state_dict(), 'modal.pkl')


        
            