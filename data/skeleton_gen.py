import os
import sys 
sys.path.append(os.path.dirname(__file__))

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


'''
For foreground skelenton and background skelenton generate
'''

def dilate(a, k=3):
    if k == 0:
        return a
    return morphology.dilation(a, morphology.square(k))

def calc_back_skelen(a):
    skelen = np.zeros_like(a)
    mask_label, label_num = measure.label(a, connectivity=1, background=1, return_num=True) # 先标记
    image_props = measure.regionprops(mask_label, cache=False) # 获得区域属性

    for li in range(label_num):
        image_prop = image_props[li]
        (min_row, min_col, max_row, max_col) = image_prop.bbox
        bool_sub = np.zeros(image_prop.image.shape)
        bool_sub[image_prop.image] = 1.0

        bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") # /255 # Lee Skeleton method
        skelen[min_row:max_row, min_col:max_col] += bool_sub_sk
        #plt.figure(figsize=(20, 10))
        #plt.subplot(1,2,1), plt.imshow(bool_sub, cmap='gray'), plt.axis("off"), plt.title('bool_sub', fontsize=24)
        #plt.subplot(1,2,2), plt.imshow(bool_sub_sk, cmap='gray'), plt.axis("off"), plt.title('bool_sub_sk', fontsize=24)
        #plt.show()

    back_skelen = dilate(skelen)
    fore_skelen =  morphology.skeletonize(a, method="lee") # /255
    back_skelen = back_skelen[:, :, np.newaxis]
    fore_skelen = fore_skelen[:, :, np.newaxis]
    skelens = np.concatenate((back_skelen, fore_skelen), axis=2)
    
    return skelens

def map_gen_skeleton(dataset_name):

    cwd = os.getcwd()

    mask_dir = cwd + "/data/" + dataset_name + "/masks/"
    map_dir = cwd + "/data/" + dataset_name + "/skea_topo_maps/"

    st = time.time()

    data_dir = mask_dir  # noqa
    label_paths = os.listdir(data_dir)
    # print('start to handle .. ', data_dir, ' -- ', len(label_paths))
    cnt = 1
    for i, lbp in enumerate(label_paths):
        if lbp.split('.')[-1] in ['png', 'tif']:
            lbp_name = lbp.split('.')[0].strip()
            # save_dir = os.path.join(os.path.abspath(os.path.dirname(data_dir)), '../skelen/')
            save_dir = map_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, lbp_name+'.npy')
            #if os.path.exists(save_path):
            #    cnt += 1
            #    print(lbp, ' already handle... skip')
            #    continue
            lbp = os.path.join(data_dir, lbp)
            #print('start to handle pic ', lbp, ' -- ', lbp_name, flush=True)
            img = io.imread(lbp)
            if np.amax(img) == 255 and len(np.unique(img)) == 2:
                img = img * 1.0 / 255.0
            #f 'snemi3d' in data or 'iron' in data:
            #    img = 1-img   # If the foreground in the image is 0, then the pixels need to be inverted.
            skelen = calc_back_skelen(img)
            #print(i, ' costs ', time.time() - st, flush=True)

            #print('skelen 0 --> min ', np.amin(skelen[:,:,0]), ' max: ', np.amax(skelen[:, :, 0]), flush=True)
            #print('skelen 1 --> min ', np.amin(skelen[:,:,0]), ' max: ', np.amax(skelen[:, :, 0]), flush=True)

            #if cnt == 1:
            #    plt.figure(figsize=(7, 10))
            #    plt.subplot(2,1,1), plt.imshow(skelen[:,:,0], cmap='gray'), plt.colorbar(), plt.title('back_skelen')
            #    plt.subplot(2,1,2), plt.imshow(skelen[:,:,1], cmap='gray'), plt.colorbar(), plt.title('fore_skelen')
            #    plt.savefig('./'+data.split('/')[0]+'_skelen.jpg')
            #    plt.show()
            #    # break
            
            skelen = np.transpose(skelen, axes=(2, 0, 1))

            matrix = np.load(save_path)
            matrix = np.concatenate((matrix, skelen), axis=0)
            C, _, _ = matrix.shape
            if C == 4:
                np.save(os.path.join(save_dir, lbp_name+'.npy'), matrix)
            else:
                print("skip")
            del skelen
            gc.collect()
            cnt += 1
        # print('handle ... ', data, ' done!')

if __name__ == "__main__":
    dataset_names = ["SNEMI3D", "DRIVE", "GlaS"]
    for dataset_name in dataset_names:
        map_gen_skeleton(dataset_name)