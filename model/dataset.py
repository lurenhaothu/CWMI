import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt

class SNEMI3DDataset(torch.utils.data.Dataset):
    def __init__(self, indices: list[int], augmentation: bool, weight_map: bool=False):
        self.indices = indices
        self.weight_map = weight_map
        cwd = os.getcwd()
        self.images_dir = cwd + "/data/images/"
        self.masks_dir = cwd + "/data/masks/"
        self.maps_dir = cwd + "/data/maps/"

        mean = [0.5053152359607174]
        std = [0.16954360899089577]

        self.norm = v2.Normalize(mean=mean, std=std)

        self.preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # scale=True: 0-255 to 0-1
        ])
        
        if augmentation:
            self.transform = v2.Compose([
                v2.RandomCrop(size=(512, 512)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip()
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        image = Image.open(self.images_dir + str(self.indices[idx]).zfill(3) + '.png')
        mask = Image.open(self.masks_dir + str(self.indices[idx]).zfill(3) + '.png')

        image = self.preprocess(image)
        mask = self.preprocess(mask)

        with torch.no_grad():
            class_weight = torch.zeros((2, 1))
            class_weight[0, 0] = torch.sum(mask == 0)
            class_weight[1, 0] = torch.sum(mask == 1)
            class_weight = class_weight * 1.0 / torch.min(class_weight)
            class_weight = torch.sum(class_weight) - class_weight

        if self.weight_map:
            w_map = np.load(self.maps_dir + str(self.indices[idx]).zfill(3) + '.npy')
            w_map = torch.tensor(w_map).unsqueeze(0).to(torch.float32)
            w_map = v2.functional.to_image(w_map)
            if self.transform != None:
                image, mask, w_map = self.transform(image, mask, w_map)
            image = self.norm(image)
            # print(image.shape, mask.shape, w_map.shape)
            return image, mask, w_map, class_weight
        else:
            if self.transform != None:
                image, mask = self.transform((image, mask))
            image = self.norm(image)
            return image, mask, torch.empty(0), class_weight
    
# test
if __name__ == "__main__":
    dataset = SNEMI3DDataset([1])
    print(len(dataset))
    img, msk = dataset[0]
    print(img.shape)
    print(msk)
    print(torch.min(msk), torch.max(msk))
    print(torch.min(img), torch.max(img))
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img.squeeze())
    axes[1].imshow(msk.squeeze())
    plt.show()
    