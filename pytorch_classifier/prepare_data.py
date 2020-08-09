# Script to prepare images for passing into the pytorch classifier
# Assumes images as produced by utils/crop_images and read by cv2.imread
# Based on TestPipeline in motorcycle_dataset

import cv2
import torch
import numpy as np


def prepare_data(img):
    # resize
    out_shape = (128, 128)
    img = cv2.resize(img, out_shape)

    # normalize
    zero_img = np.zeros(out_shape)
    norm_img = cv2.normalize(img, zero_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = norm_img / 255.0

    # convert to tensor
    tens = np.transpose(norm_img, (2, 1, 0))
    tens = torch.from_numpy(tens).float()

    # add dimension
    result = torch.unsqueeze(tens, 0)

    return result
