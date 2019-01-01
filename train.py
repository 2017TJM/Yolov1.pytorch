from model.head import DetectionHead
from model.yolo import Yolo
from data.shape_dataset import ShapeDataset
from data.preprocess import YoloAugmentation
from data import detection_collate
from util.vis_util import plot_bbox, plot_grid, plot_image
from util.bbox_util import xywh_to_xyxy
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

    model = Yolo(DetectionHead(C=cfg.C, B=cfg.B)).to(device)
    if os.path.exists('model.pkl'):
        print('Loading model...')
        model.load_state_dict(t.load('model.pkl'))

    trainer = SGD(model.parameters(), lr=0.01, momentum=0.8)
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
            loss, coord_loss, conf_loss, class_loss = criteria(output, bboxes, labels)
            loss.backward()
            trainer.step()
            train_loss += loss.cpu().item()
            print('Loss: %.3f coord_loss: %.3f  conf_loss: %.3f  class_loss: %.3f' % (train_loss / i , coord_loss, conf_loss, class_loss))
        print('Epoch %d  loss: %.3f' % (e, train_loss / i))
        t.save(model.state_dict(), 'model.pkl')


    # import cupy as cp
    # import cv2
    # # from util.nms import non_maximum_suppression
    
    # filename = './dataset/shape/images/cfe8b649-11c6-4ae8-8383-25ad4ff17e84.png'
    # img = cv2.imread(filename)
    # img = cv2.resize(img, (448, 448))
    

    # from util.bbox_util import IOU

    # def nms(bboxes, scores, overlap=0.5, top_k=200):
    #     keep = scores.new(scores.shape[0]).zero_().long()
    #     desend_scores, indices = t.sort(scores, descending=True)
    #     indices = indices[:min(indices.shape[0], top_k)]

    #     count = 0
    #     while indices.numel() > 0:
    #         idx = indices[0]
    #         keep[count] = idx
    #         count += 1

    #         indices = indices[1:]
    #         ious = IOU(bboxes[idx].view(1, 4), bboxes[indices, :])
    #         mask = ious[0, :] < overlap
    #         indices = indices[mask]
    #     return keep[:count]

    # model.train()
    # with t.no_grad():
    #     output = model(t.from_numpy(img).permute(2, 0, 1).view(1, 3, 448, 448).float())
    #     batch_size = output.shape[0]
    #     index1 = cfg.S*cfg.S*cfg.C
    #     index2 = index1 + cfg.S*cfg.S*cfg.B
    #     # [batch_size, S*S, C]
    #     class_prob = output[:, :index1].view(batch_size, cfg.S*cfg.S, cfg.C)
    #     # [batch_size, S*S, B]
    #     conf = output[:, index1:index2].view(-1)
    #     # [batch_size, S*S*B, 4]
    #     bboxes = output[:, index2:].view(cfg.S*cfg.S, cfg.B, 4)
    #     bboxes = xywh_to_xyxy(bboxes).view(-1, 4)
    #     # plot_bbox(img, bboxes, [0]*bboxes.shape[0], cfg.shape_bbox_colors)
    #     keep = nms(bboxes, conf, overlap=0.3, top_k=200)
    #     keep_bboxes = bboxes[keep, :].int()
    #     print(keep_bboxes)
    #     plot_bbox(img, keep_bboxes, [1]*keep_bboxes.shape[0], cfg.shape_bbox_colors)
            