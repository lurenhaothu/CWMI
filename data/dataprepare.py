import PIL.Image
import tifffile as tif
import os
import PIL
import skimage
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()

images_dir = cwd + "/data/train-input.tif"
masks_dir = cwd + "/data/train-labels.tif"

images_output_dir = cwd + "/data/images/"
os.makedirs(images_output_dir, exist_ok=True)
masks_output_dir = cwd + "/data/masks/"
os.makedirs(masks_output_dir, exist_ok=True)

images = tif.TiffFile(images_dir)
masks = tif.TiffFile(masks_dir)

len = len(images.pages)

imageMean = np.zeros(len)
imageStd = np.zeros(len)
foreCount = 0
backCount = 0
H, W = 0, 0

for i in range(len):
    image = images.pages[i].asarray()
    mask = masks.pages[i].asarray()

    imageMean[i] = np.mean(image / 255)
    imageStd[i] = np.std(image / 255)

    image = PIL.Image.fromarray(image)
    image.save(images_output_dir + str(i).zfill(3) + '.png')

    mask[skimage.segmentation.find_boundaries(mask, connectivity=1, background=0)] = 0
    mask[mask != 0] = 1
    mask = 1 - mask

    H, W = mask.shape
    curSum = np.sum(mask)
    foreCount += curSum
    backCount += (H * W) - curSum

    mask = PIL.Image.fromarray((mask * 255).astype(np.uint8))

    #plt.imshow(mask)
    #plt.show()

    mask.save(masks_output_dir + str(i).zfill(3) + '.png')

mean = np.mean(imageMean)
std = np.sqrt(np.sum(np.power(imageStd, 2) + np.power(imageMean - mean, 2)) / len)
print(len, mean, std, foreCount, backCount)