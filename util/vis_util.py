import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_grid(imgs, n_row=3, n_col=3):
    _, figs = plt.subplots(n_row, n_col)

    for i in range(n_row):
        for j in range(n_col):
            figs[i][j].imshow(imgs[i*n_row + j])
            figs[i][j].axis('off')
    plt.show()

def plot_bbox(img, bboxes, labels, label_to_color):
    # Attention!!!!
    img = img.copy()
    for i in range(len(bboxes)):
        cv2.rectangle(img, (bboxes[i, 1], bboxes[i, 0]), (bboxes[i, 3], bboxes[i, 2]), label_to_color[labels[i]], 1)

    plt.axis('off')
    plt.imshow(img)
    plt.show()
    return img

