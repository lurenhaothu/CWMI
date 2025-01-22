from PIL import Image
import os
import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

w0 = 10
sigma = 5

def get_dis_map(index, mask_label):
    item_map = (mask_label != index).astype(int)
    return index, distance_transform_edt(item_map)

def get_first_2(j, chunk):
    return j, np.sum(np.partition(chunk, 2, axis=0)[0:2, :, :], axis=0)

def map_gen_unet(dataset_name):
    cwd = os.getcwd()
    mask_dir = cwd + "/data/" + dataset_name + "/masks/"
    map_dir = cwd + "/data/" + dataset_name + "/unet_maps/"

    os.makedirs(map_dir, exist_ok=True)

    length = len(os.listdir(mask_dir))

    for i in range(length):
        t = time.time()

        mask = np.array(Image.open(mask_dir + str(i).zfill(3) + ".png"))
        mask = (mask == 255).astype(int)
        mask_label, num = label(mask, background=1, connectivity=1, return_num=True)

        H, W = mask.shape

        dis_map = np.zeros((num, H, W))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_dis_map, j, mask_label) for j in range(1, num + 1)]

            for future in futures:
                j, dis_map_j = future.result()
                dis_map[j - 1,:,:] = dis_map_j

        if num > 2:
            sum_dis_map = np.zeros((H, W))

            #sum_dis_map = np.sum(np.partition(dis_map * mask, 2, axis=0)[0:2, :, :], axis=0)
            
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_first_2, j, dis_map[:, j:j+1, :]) for j in range(1024)]

                for future in futures:
                    j, chunk = future.result()
                    sum_dis_map[j:j+1, :] = chunk

            weight_map = w0 * np.exp( - np.power(sum_dis_map , 2) / 2 / sigma / sigma) * mask
        
        else:
            weight_map = w0 * np.exp( - np.power(2 * dis_map[0,...] , 2) / 2 / sigma / sigma) * mask

        #plt.imshow(weight_map)
        #plt.show()
        #break
        np.save(map_dir + str(i).zfill(3) + ".npy", weight_map)

        print('finished ' + str(i) + ' time: ' + str(time.time() - t))

if __name__ == "__main__":
    dataset_names = ["GlaS"]
    for dataset_name in dataset_names:
        map_gen_unet(dataset_name)