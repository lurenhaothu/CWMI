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

def map_2018kdsb(pred: np.ndarray, mask: np.ndarray, bg_value = 1) -> float:
    """
    he metric is referenced from 2018 kaggle data science bowl: 
    https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    :param pred: prediction
    :param mask: ground truth
    :param bg_value: background value used for label function
    :return: map_score
    """
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    fp = np.zeros(10)
    fn = np.zeros(10)
    if np.amax(mask) == 255:
        pred = pred / 255
        mask = mask / 255
    # optimize border of image
    pred[0, :] = bg_value; pred[:, 0] = bg_value; pred[-1, :] = bg_value; pred[:, -1] = bg_value
    mask[0, :] = bg_value; mask[:, 0] = bg_value; mask[-1, :] = bg_value; mask[:, -1] = bg_value
        
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))   # Get all labels intersecting with it
        # 对与其相交的的所有mask label计算iou，后取其最值  Calculate IOU for all mask labels intersecting with it, then take the maximum value
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:   # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label  If a label in 'pred' does not have a corresponding label in the mask, then move on to the next label.
            continue
        
        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        
        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值   Assign 'TP' according to the maximum value
        # 此处基于一个重要理论：对于一个预测的晶粒，真实的晶粒有且仅有一个晶粒与其iou>0.5   Based on an important assumption: for a predicted grain, there is exactly one true grain with which its IOU is greater than 0.5.
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    fp = num_pred - tp 
    fn = num_mask - tp
    map_score = np.average(tp/(tp + fp + fn))
    return map_score

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

def get_betti_own(x, is_show=False, filter_small_holes=False):  # binary_image  foreground 1， background 0
    # The 0th Betti number 𝑏0 represents the number of connected components, is equivalent to counting the number of connected components in the foreground.
    # The 1st Betti number 𝑏1 represents the number of holes, is equivalent to counting the number of connected components in the background.
    # the 2nd Betti number 𝑏2 represents the number of cavities. 
    mask_label_0, label_num_0 = label(x, connectivity=2, background=0, return_num=True) # label foreground connected regions
    mask_label_1, label_num_1 = label(x, connectivity=1, background=1, return_num=True) # label background connected regions
    if is_show:   # show case
        plt.figure(figsize=(15, 5))
        plt.subplot(1,3,1), plt.imshow(x, cmap='plasma'), plt.axis("off")
        plt.subplot(1,3,2), plt.imshow(mask_label_0, cmap='plasma'), plt.axis("off")
        plt.subplot(1,3,3), plt.imshow(mask_label_1, cmap='plasma'),  plt.axis("off")
        plt.show()
    
    if filter_small_holes:
        label_num_0_filter = filter_small_holes_func(mask_label_0, label_num_0, is_fore=True)
        label_num_1_filter = filter_small_holes_func(mask_label_1, label_num_1, is_fore=False)
        return label_num_0_filter, label_num_1_filter
    return label_num_0, label_num_1


def compute_bettis_own(pred, label, filter_small_holes=False):
    label_betti0, label_betti1 = get_betti_own(label)
    pred_betti0, pred_betti1 = get_betti_own(pred, filter_small_holes=filter_small_holes)

    betti0_error = abs(label_betti0-pred_betti0)
    betti1_error = abs(label_betti1-pred_betti1)
    return betti0_error+betti1_error # , betti0_error, betti1_error

# Post processing
def prun(image, kernel_size):
    """
    Remove small forks
    """
    label_map, num_label = label(image, connectivity=1, background=1, return_num=True)
    result = np.zeros(label_map.shape)
    for i in range(1, num_label + 1):
        tmp = np.zeros(label_map.shape)
        tmp[label_map == i] = 1
        D_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dil = cv2.dilate(tmp, D_kernel)
        dst = cv2.erode(dil, D_kernel)
        result[dst == 1] = 255
    result = 255 - result
    result[result == 255] = 1
    result = np.uint8(result)
    return result


def post_process_label(in_img: np.ndarray, prun=False) -> np.ndarray:
    out_img = morphology.skeletonize(in_img, method="lee")
    out_img = morphology.dilation(out_img, morphology.square(3))
    if prun:
        out_img = prun(out_img, 4)  # 5
    return out_img


def post_process_output(in_img: np.ndarray, prun=False) -> np.ndarray:
    out_img = morphology.dilation(in_img, morphology.square(3))  # 2  3
    out_img = morphology.skeletonize(out_img, method="lee")
    out_img = morphology.dilation(out_img, morphology.square(3))  # 5  3
    if prun:
        out_img = prun(out_img, 4)  # 5
    return out_img

if __name__ == "__main__":
    # test vi
    mask = (np.random.rand(512, 512) > 0.5).astype(int)
    pred = (np.random.rand(512, 512) > 0.5).astype(int)

    t = time.time()

    print(vi(mask, pred))
    print(miou(pred, mask))
    print(mdice(pred, mask))
    print(map_2018kdsb(pred, mask))
    print(ari(pred, mask))
    print(compute_bettis_own(pred, mask))

    print("cal time = " + str(time.time() - t))