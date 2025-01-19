from PIL import Image
import os
import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

cwd = os.getcwd()
mask_dir = cwd + "/data/masks/"
map_dir = cwd + "/data/maps/"
os.makedirs(map_dir, exist_ok=True)

w0 = 10
sigma = 5
single = False

def get_dis_map(index, mask_label):
    item_map = (mask_label != index).astype(int)
    return index, distance_transform_edt(item_map)

def get_first_2(j, chunk):
    return j, np.sum(np.partition(chunk, 2, axis=0)[0:2, :, :], axis=0)

if single:
    single_arr = np.zeros((100, 1024, 1024))

for i in range(100):
    t = time.time()

    mask = np.array(Image.open(mask_dir + str(i).zfill(3) + ".png"))
    mask = (mask == 255).astype(int)
    mask_label, num = label(mask, background=1, connectivity=1, return_num=True)

    dis_map = np.zeros((num, 1024, 1024))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_dis_map, j, mask_label) for j in range(1, num + 1)]

        for future in futures:
            j, dis_map_j = future.result()
            dis_map[j - 1,:,:] = dis_map_j

    sum_dis_map = np.zeros((1024, 1024))

    #sum_dis_map = np.sum(np.partition(dis_map * mask, 2, axis=0)[0:2, :, :], axis=0)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_first_2, j, dis_map[:, j:j+1, :]) for j in range(1024)]

        for future in futures:
            j, chunk = future.result()
            sum_dis_map[j:j+1, :] = chunk

    weight_map = w0 * np.exp( - np.power(sum_dis_map , 2) / 2 / sigma / sigma) * mask

    #plt.imshow(weight_map)
    #plt.show()
    #break
    if not single:
        np.save(map_dir + str(i).zfill(3) + ".npy", weight_map)

    if single:
        single_arr[i,:,:] = weight_map

    print('finished ' + str(i) + ' time: ' + str(time.time() - t))

if single:
    np.save(map_dir + "maps.npy", single_arr)

