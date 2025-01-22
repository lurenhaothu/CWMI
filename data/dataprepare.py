from PIL import Image
import tifffile as tif
import os
import PIL
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

cwd = os.getcwd()

def data_prepare_SNEMI3D():

    images_dir = cwd + "/data/SNEMI3D/train-input.tif"
    masks_dir = cwd + "/data/SNEMI3D/train-labels.tif"

    images_output_dir = cwd + "/data/SNEMI3D/images/"
    os.makedirs(images_output_dir, exist_ok=True)
    masks_output_dir = cwd + "/data/SNEMI3D/masks/"
    os.makedirs(masks_output_dir, exist_ok=True)

    images = tif.TiffFile(images_dir)
    masks = tif.TiffFile(masks_dir)

    length = len(images.pages)

    imageMean = np.zeros(length)
    imageStd = np.zeros(length)

    for i in range(length):
        image = images.pages[i].asarray()
        mask = masks.pages[i].asarray()

        imageMean[i] = np.mean(image / 255)
        imageStd[i] = np.std(image / 255)

        image = PIL.Image.fromarray(image)
        image.save(images_output_dir + str(i).zfill(3) + '.png')

        mask[skimage.segmentation.find_boundaries(mask, connectivity=1, background=0)] = 0
        mask[mask != 0] = 1
        mask = 1 - mask

        mask = Image.fromarray((mask * 255).astype(np.uint8))

        #plt.imshow(mask)
        #plt.show()

        mask.save(masks_output_dir + str(i).zfill(3) + '.png')

    mean = np.mean(imageMean)
    std = np.sqrt(np.sum(np.power(imageStd, 2) + np.power(imageMean - mean, 2)) / length)
    print("finish processing SNEMI3D dataset", " len: ", length, " mean: ", mean, " std: ", std)

    dataset_metadata = pd.DataFrame({"Name": ["SNEMI3D"], "len": [length], "mean": [mean], "std": [std]})
    
    dataset_metadata.to_csv(cwd + "/data/SNEMI3D/dataset_metadata.csv", index=False)

def data_prepare_Drive():

    test_images_dir = cwd + "/data/DRIVE/Drive_source/test/images/"
    test_masks_dir = cwd + "/data/DRIVE/Drive_source/test/1st_manual/"

    train_images_dir = cwd + "/data/DRIVE/Drive_source/training/images/"
    tain_masks_dir = cwd + "/data/DRIVE/Drive_source/training/1st_manual/"

    images_output_dir = cwd + "/data/DRIVE/images/"
    os.makedirs(images_output_dir, exist_ok=True)
    masks_output_dir = cwd + "/data/DRIVE/masks/"
    os.makedirs(masks_output_dir, exist_ok=True)

    images_paths = {}
    for p in os.listdir(test_images_dir):
        images_paths[int(p[:2])] = test_images_dir + p
    for p in os.listdir(train_images_dir):
        images_paths[int(p[:2])] = train_images_dir + p
    masks_paths = {}
    for p in os.listdir(test_masks_dir):
        masks_paths[int(p[:2])] = test_masks_dir + p
    for p in os.listdir(tain_masks_dir):
        masks_paths[int(p[:2])] = tain_masks_dir + p

    length = len(images_paths)

    imageMean = np.zeros((length, 3))
    imageStd = np.zeros((length, 3))

    for i in range(length):
        image = Image.open(images_paths[i + 1])
        image_np = np.array(image)
        mask = Image.open(masks_paths[i + 1])
        mask_np = np.array(mask)

        H, W, C = image_np.shape
        pad_H = math.ceil(H / 32) * 32 - H
        pad_W = math.ceil(W / 32) * 32 - W

        image_np = np.pad(image_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2), (0, 0)), mode='edge')
        mask_np = np.pad(mask_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2)), mode='edge')

        imageMean[i, :] = np.mean(image_np / 255, axis=(0, 1))
        imageStd[i, :] = np.std(image_np / 255, axis=(0, 1))

        Image.fromarray(image_np).save(images_output_dir + str(i).zfill(3) + '.png')

        mask = Image.fromarray(mask_np.astype(np.uint8))

        mask.save(masks_output_dir + str(i).zfill(3) + '.png')

    mean = np.mean(imageMean, axis=0)
    std = np.sqrt(np.sum(np.power(imageStd, 2) + np.power(imageMean - mean, 2), axis=0) / length)
    print("finish processing DRIVE dataset", " len: ", length, " mean: ", mean, " std: ", std)

    dataset_metadata = pd.DataFrame({"Name": ["DRIVE " + c for c in "RGB"], "mean": mean.tolist(), "std": std.tolist()})
    
    dataset_metadata.to_csv(cwd + "/data/DRIVE/dataset_metadata.csv", index=False)

def data_prepare_GlaS():

    images_dir = cwd + "/data/GlaS/Warwick_QU_Dataset/"

    images_output_dir = cwd + "/data/GlaS/images/"
    os.makedirs(images_output_dir, exist_ok=True)
    masks_output_dir = cwd + "/data/GlaS/masks/"
    os.makedirs(masks_output_dir, exist_ok=True)

    all_paths = [(images_dir + p) for p in os.listdir(images_dir) if p.endswith('.bmp')]
    images_paths = [p for p in all_paths if not p.endswith('anno.bmp')]
    masks_paths = [p[:-4] + '_anno.bmp' for p in images_paths]

    length = len(images_paths)

    imageMean = np.zeros((length, 3))
    imageStd = np.zeros((length, 3))

    for i in range(length):
        image = Image.open(images_paths[i])
        image_np = np.array(image)
        mask = Image.open(masks_paths[i])
        mask_np = np.array(mask)

        H, W, C = image_np.shape
        pad_H = math.ceil(H / 32) * 32 - H
        pad_W = math.ceil(W / 32) * 32 - W

        image_np = np.pad(image_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2), (0, 0)), mode="constant", constant_values=255)
        mask_np = np.pad(mask_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2)), mode="constant", constant_values=0)

        mask_np[mask_np != 0] = 1
        mask_np = 1 - mask_np

        imageMean[i, :] = np.mean(image_np / 255, axis=(0, 1))
        imageStd[i, :] = np.std(image_np / 255, axis=(0, 1))

        Image.fromarray(image_np).save(images_output_dir + str(i).zfill(3) + '.png')

        mask = Image.fromarray((mask_np * 255).astype(np.uint8))

        mask.save(masks_output_dir + str(i).zfill(3) + '.png')

    mean = np.mean(imageMean, axis=0)
    std = np.sqrt(np.sum(np.power(imageStd, 2) + np.power(imageMean - mean, 2), axis=0) / length)
    print("finish processing GlaS dataset", " len: ", length, " mean: ", mean, " std: ", std)

    dataset_metadata = pd.DataFrame({"Name": ["GlaS " + c for c in "RGB"], "mean": mean.tolist(), "std": std.tolist()})
    
    dataset_metadata.to_csv(cwd + "/data/GlaS/dataset_metadata.csv", index=False)

def select_from_mass_road():
    mask_dir = cwd + "/data/mass_road/mass_road_source/tiff/train_labels/"
    image_dir = cwd + "/data/mass_road/mass_road_source/tiff/train/"
    count = []
    for mask_file_name in os.listdir(mask_dir):
        if not mask_file_name.endswith(".tif"):
            continue
        mask = np.array(Image.open(mask_dir + mask_file_name))
        image = np.array(Image.open(image_dir + mask_file_name + 'f'))
        mask_of_blank_region = ((image[:,:,0] == 255) & (image[:,:,1] == 255) & (image[:,:,2] == 255)).astype(int)
        count.append((np.sum(np.array(mask) * (1 - mask_of_blank_region) / 255), mask_file_name))
    count.sort(reverse=True)
    file_list = [file_name for c, file_name in random.sample(count[:300], 120)]
    file_list_pd = pd.DataFrame({"File_name": file_list})
    file_list_pd.to_csv(cwd + "/data/mass_road/selected_files.csv", index=False)
    return file_list

def data_prepare_mass_road(file_list):
    image_dir = cwd + "/data/mass_road/mass_road_source/tiff/train/"
    mask_dir = cwd + "/data/mass_road/mass_road_source/tiff/train_labels/"

    images_output_dir = cwd + "/data/mass_road/images/"
    os.makedirs(images_output_dir, exist_ok=True)
    masks_output_dir = cwd + "/data/mass_road/masks/"
    os.makedirs(masks_output_dir, exist_ok=True)

    length = len(file_list)

    imageMean = np.zeros((length, 3))
    imageStd = np.zeros((length, 3))

    for i in range(length):
        image = Image.open(image_dir + file_list[i] + 'f')
        image_np = np.array(image)
        mask = Image.open(mask_dir + file_list[i])
        mask_np = np.array(mask)

        H, W, C = image_np.shape
        pad_H = math.ceil(H / 32) * 32 - H
        pad_W = math.ceil(W / 32) * 32 - W

        image_np = np.pad(image_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2), (0, 0)), mode='edge')
        mask_np = np.pad(mask_np, ((pad_H // 2, pad_H - pad_H // 2), (pad_W // 2, pad_W - pad_W // 2)), mode='edge')

        imageMean[i, :] = np.mean(image_np / 255, axis=(0, 1))
        imageStd[i, :] = np.std(image_np / 255, axis=(0, 1))

        Image.fromarray(image_np).save(images_output_dir + str(i).zfill(3) + '.png')

        mask = Image.fromarray((mask_np).astype(np.uint8))

        mask.save(masks_output_dir + str(i).zfill(3) + '.png')

    mean = np.mean(imageMean, axis=0)
    std = np.sqrt(np.sum(np.power(imageStd, 2) + np.power(imageMean - mean, 2), axis=0) / length)
    print("finish processing mass_road dataset", " len: ", length, " mean: ", mean, " std: ", std)

    dataset_metadata = pd.DataFrame({"Name": ["mass_road " + c for c in "RGB"], "mean": mean.tolist(), "std": std.tolist()})
    
    dataset_metadata.to_csv(cwd + "/data/mass_road/dataset_metadata.csv", index=False)

if __name__ == "__main__":
    # data_prepare_SNEMI3D()
    # data_prepare_Drive()
    data_prepare_GlaS()
    #file_list = select_from_mass_road()
    #df = pd.read_csv(cwd + "/data/mass_road/selected_files.csv")
    #file_list = df["File_name"].tolist()
    #data_prepare_mass_road(file_list)
    pass