import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import Tensor
import SimpleITK as sitk


def plot_result(model_dir, H_train, H_validation, H_train_name, H_validation_name, labelname):
    PLOT_PATH = os.path.sep.join([model_dir, H_train_name + "_" + H_validation_name + "plot.png"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H_train, label=H_train_name)
    plt.plot(H_validation, label=H_validation_name)
    plt.title(H_train_name + "," + H_validation_name + " on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel(labelname)
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)


def save_images3d(pdmask: Tensor, gtmask: Tensor, size, path, pixelvalue=255.):
    images = pdmask.detach().cpu().squeeze().numpy()
    gtmask = gtmask.detach().cpu().squeeze().numpy()
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    cv2.imwrite(path + "pdmask.bmp", np.clip(merge_img * pixelvalue, 0, 255).astype('uint8'))

    merge_mask = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(gtmask):
        i = idx % size[1]
        j = idx // size[1]
        merge_mask[j * h:j * h + h, i * w:i * w + w] = image
    cv2.imwrite(path + "gtmask.bmp", np.clip(merge_mask * pixelvalue, 0, 255).astype('uint8'))


def save_images2d(pdmask: Tensor, gtmask: Tensor, path, pixelvalue=255.):
    pdmask = pdmask.detach().cpu().squeeze().numpy()
    gtmask = gtmask.detach().cpu().squeeze().numpy()
    if np.max(gtmask) == 1:
        pdmask[pdmask > 0.5] = 1
        pdmask[pdmask < 0.5] = 0
    cv2.imwrite(path + "pdmask.bmp", np.clip(pdmask * pixelvalue, 0, 255).astype('uint8'))
    cv2.imwrite(path + "gtmask.bmp", np.clip(gtmask * pixelvalue, 0, 255).astype('uint8'))
