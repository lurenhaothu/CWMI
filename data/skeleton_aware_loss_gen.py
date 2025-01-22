import os
import sys 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import gc
import math
import numpy as np
import cv2
import torch
import time
from skimage import measure
from skimage import io
import scipy.ndimage as ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor


# from model.utils.utils_overlap import OverlapTile


class SkeletonAwareWeight():
    """
    Skeleton Aware Weight
    
    """
    def __init__(self, eps = 1e-20):
        """
        At this time, the weight is only suited to binary segmentation, so class_num = 2
        """
        self._class_num = 2
        self._eps = eps
        self.method=None
        # self.overlap_tile = OverlapTile()  # speed up

    
    def _get_weight(self, mask: np.ndarray, method=None, single_border=False) -> np.ndarray:
        """
        Get skeleton aware weight map
        :param mask: binary gt mask with shape (H, W)
        :return weight:  weight map with shape (H, W, 2)
        """
        self.method = method
        # Get class weight for two channels
        weight = np.zeros((mask.shape[0], mask.shape[1], 2))
        class_weight = np.zeros((self._class_num, 1))
        for class_idx in range(self._class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight        
        
        
        # Get weight for each channel
        for class_idx in range(self._class_num):
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == class_idx] = 1.0
            dis_trf = ndimage.distance_transform_edt(temp_mask)
            
            st = time.time()
            if class_idx == 1: 
                # Get weight for border
                if single_border:
                    temp_weight = 1.0
                else:
                    label_map, label_num = measure.label(temp_mask, connectivity=1, background=1, return_num=True)

                    t = time.time()

                    temp_weight = self._get_border_weight(class_weight[class_idx, 0], temp_mask, dis_trf, label_map, label_num)

                    print("boarder weight time: ", time.time() - t)
                    ''' # speed up
                    crop_images = self.overlap_tile.crop(temp_mask)
                    temp = []
                    for crop_mask in crop_images:
                        label_map, label_num = measure.label(crop_mask, connectivity=1, background=1, return_num=True)
                        temp_weight = self._get_border_weight(class_weight[class_idx, 0], crop_mask, dis_trf, label_map, label_num)
                        temp.append(temp_weight)
                    temp_weight = self.overlap_tile.stitch(temp)
                    '''
            else:
                # Get weight for objects
                label_map, label_num = measure.label(temp_mask, connectivity=1, background=0, return_num=True)

                t = time.time()

                temp_weight = self._get_object_weight(class_weight[class_idx, 0], dis_trf, label_map, label_num)

                print("object weight time: ", time.time() - t)

            ed = time.time()
            print("class:{}, time {:.4f}".format(class_idx, ed - st))
            weight[:, :, class_idx] = temp_weight * temp_mask
        return weight
    
    def get_dis_map(self, index, mask_label):
        item_map = (mask_label != index).astype(int)
        return index, ndimage.distance_transform_edt(item_map)
    
    def get_first_2(self, j, chunk):
        return j, np.partition(chunk, 2, axis=2)[:, :, 0:2]
    
    def _get_border_weight(self, wc: float, mask: np.ndarray, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get border weight
        :param wc: class weight of border channel
        :param mask: real mask with shape (H,W), the border pixels equal 0 and the object pixels equal 1
        :param dis_trf: distance transform of border (shape (H,W)), it means the distance of each border pixel to the nearest object
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of border with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        # image_props = measure.regionprops(label_map, cache=False)
        h, w = mask.shape[:2]
        min_dis = np.ones((h, w, label_num))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_dis_map, j, label_map) for j in range(1, label_num + 1)]

            for future in futures:
                j, dis_map_j = future.result()
                min_dis[:,:, j - 1] = dis_map_j
        
        '''
        # For each connected region, calculate the closest distance from other pixels to this region.
        for label_idx in range(label_num): 
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0
            # mask this connected region by set its value to 0 
            mask_label = np.ones_like(mask)
            mask_label[min_row: max_row, min_col: max_col] -= bool_sub
            # Get the distance transform to masked connect component
            min_dis[:,:,label_idx] = ndimage.distance_transform_edt(mask_label) * mask
        '''

        min_dis_2 = np.zeros((h, w, 2))

        if label_num > 2:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.get_first_2, j, min_dis[j:j+1, :, :]) for j in range(h)]

                for future in futures:
                    j, chunk = future.result()
                    min_dis_2[j:j+1, :, :] = chunk
        elif label_num == 2:
            min_dis_2 = min_dis
        else:
            min_dis_2[:,:,0] = min_dis[:,:,0]
            min_dis_2[:,:,1] = min_dis[:,:,0]
            
        # Calculate for each pixel the distances to the first nearest(d1) and second nearest(d2) connected regions
        # min_dis.sort(axis=2)
        # min_dis = min_dis[:, :, 0:2]
        min_dis = min_dis_2
        # Calculate weight map by d1 and d2 for each pixel
        max_dis_trf = np.amax(dis_trf)
        weight =  1.0 + ((2 * max_dis_trf - min_dis[:, :, 0] - min_dis[:, :, 1] + self._eps) / (2 * max_dis_trf + self._eps))

        weight[weight < 0] = 0.0
        return weight

    def _get_object_weight(self, wc: float, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get object weight
        :param wc: class weight of border channel
        :param dis_trf: distance transform of object (shape (H,W)), it means the distance of each pixel to the nearest border
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of object with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        
        # For each connect component, calculate its weight by its skeleton
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") # /255 # Lee Skeleton method
            if np.count_nonzero(bool_sub_sk == 1.0) == 0:
                # If there is no skelenton pixel, continue
                continue
            # Get the distance transform of skeleton pixel
            dis_trf_sk_sub = dis_trf[min_row: max_row, min_col: max_col] * bool_sub_sk

            # Get the distance transform to skeleton pixel 
            indices = np.zeros(((np.ndim(bool_sub_sk),) + bool_sub_sk.shape), dtype=np.int32)
            dis_trf_to_sk = ndimage.distance_transform_edt(1-bool_sub_sk, return_indices=True, indices=indices)

            h, w = bool_sub.shape[:2]
            dis_sk_map = np.ones((h, w, 2))
            dis_sk_map[:, :, 0] = dis_trf_to_sk  # d0 
            dis_sk_map[:, :, 1] = dis_trf_sk_sub[indices[0, :, :], indices[1, :, :]]   # d1 

            # Rectify, enusre d0 <= d1, d0: the distance of pixel to nearest skeleton pixel, d1: the distance d1 of nearest skeleton pixel to border
            dis_sk_map[:, :, 0][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]] = dis_sk_map[:, :, 1][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]]    
 
            weight_sub = 1-(dis_sk_map[:, :, 0] / (dis_sk_map[:, :, 1] + self._eps))
            
            weight[min_row: max_row, min_col: max_col] += weight_sub * bool_sub

        
        return weight


def map_gen_skea_topo(dataset_name):

    cwd = os.getcwd()

    mask_dir = cwd + "/data/" + dataset_name + "/masks/"
    map_dir = cwd + "/data/" + dataset_name + "/skea_topo_maps/"
    weight_fuc = SkeletonAwareWeight()
    st = time.time()

    cwd = os.getcwd()

    data_dir = mask_dir
    label_paths = os.listdir(data_dir)
    # print('start to handle .. ', data_dir, ' -- ', len(label_paths))
    cnt = 0
    for i, lbp in enumerate(label_paths):
        if lbp.split('.')[-1] in ['png', 'tif']:
            lbp_name = lbp.split('.')[0].strip()
            lbp = os.path.join(data_dir, lbp)
            # print('start to handle pic ', lbp, ' -- ', lbp_name)
            img = io.imread(lbp)
            if np.amax(img) == 255 and len(np.unique(img)) == 2:
                img = img * 1.0 / 255.0

            single_border = False
            #if 'iron' in data or 'mass_road' in data:
            #    single_border = True   # evenly placed objects
            #if 'snemi3d' in data or 'iron' in data:
            #    img = 1-img   # If the foreground pixels in the image are 0, then the pixels need to be inverted.
            weight = weight_fuc._get_weight(img, single_border=single_border)
            #print(i, ' costs ', time.time() - st)
            #print('weight 0 --> min ', np.amin(weight[:,:,0]), ' max: ', np.amax(weight[:, :, 0]))
            #print('weight 1 --> min ', np.amin(weight[:,:,1]), ' max: ', np.amax(weight[:, :, 1]))

            #if cnt == 1:
            #    plt.figure(figsize=(100, 100))
            #    plt.subplot(1,2,1), plt.imshow(weight[:,:,0], cmap='plasma'), plt.title('weight_0')
            #    plt.subplot(1,2,2), plt.imshow(weight[:,:,1], cmap='plasma'), plt.title('weight_1')
            #    # plt.savefig('./weight_'+data.split('/')[0]+'.jpg')
            #    plt.show()
            #    #break
            
            save_dir = map_dir  # noqa
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            weight = np.transpose(weight, axes=(2, 0, 1))
            np.save(os.path.join(save_dir, lbp_name+'.npy'), weight)
            del weight
            gc.collect()
            cnt += 1

if __name__ == "__main__":
    dataset_names = ["GlaS"]
    for dataset_name in dataset_names:
        map_gen_skea_topo(dataset_name)