import skimage.metrics as metrics
from skimage.measure import label
import numpy as np
import time
import cv2
from skimage import morphology
import math
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

def hausdorff_distance(pred: np.array, mask: np.array):
    return metrics.hausdorff_distance(pred, mask, method='modified')


def vi(pred: np.array, mask: np.array):
    mask_label = label(mask, background=1, connectivity=1)
    pred_label = label(pred, background=1, connectivity=1)
    merger_error, split_error = metrics.variation_of_information(mask_label, pred_label, ignore_labels=[0])
    # merger_error, split_error = metrics.variation_of_information(pred_label, mask_label, ignore_labels=[0])

    vi = merger_error + split_error
    if math.isnan(vi):
        return 10
    return vi

def miou(pred: np.ndarray, mask: np.ndarray, n_cl=2) -> float:
    """
    # mean iou, intersection over union
    :param pred: prediction
    :param mask: ground truth
    :param n_cl: class number
    :return: miou_score
    """
    if np.amax(mask) == 255 and n_cl == 2:
        pred = pred / 255
        mask = mask / 255
    iou = 0
    for i_cl in range(0, n_cl):
        intersection = np.count_nonzero(mask[pred == i_cl] == i_cl)
        union = np.count_nonzero(mask == i_cl) + np.count_nonzero(pred == i_cl) - intersection
        iou += intersection / union
    miou_score = iou / n_cl
    return -miou_score

def mdice(pred: np.ndarray, mask: np.ndarray, n_cl=2) -> float:
    """
    :param pred: prediction
    :param mask: ground truth
    :param n_cl: class number
    :return: mdice_score
    """
    if np.amax(mask) == 255 and n_cl == 2:
        pred = pred / 255
        mask = mask / 255
    dice = 0
    for i_cl in range(0, n_cl):
        intersection = np.count_nonzero(mask[pred == i_cl] == i_cl)
        area_sum = np.count_nonzero(mask == i_cl) + np.count_nonzero(pred == i_cl)
        dice += 2 * intersection / area_sum
    mdice_score = dice / n_cl
    return mdice_score

def ari(in_pred: np.ndarray, in_mask: np.ndarray, bg_value = 1) -> float:
    pred = in_pred.copy()
    mask = in_mask.copy()
    if np.amax(mask) == 255:
        pred = pred / 255
        mask = mask / 255
    
    label_pred, _ = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, _ = label(mask, connectivity=1, background=bg_value, return_num=True)    
    #adjust_RI = ev.adj_rand_index(label_pred, label_mask)
    # already imported
    adjust_RI = adjusted_rand_score(label_pred.flatten(), label_mask.flatten())
    return adjust_RI