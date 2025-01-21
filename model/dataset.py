import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, indices: list[int], augmentation: bool, weight_map_name=None):
        self.indices = indices
        cwd = os.getcwd()
        self.images_dir = cwd + "/data/" + dataset_name + "/images/"
        self.masks_dir = cwd + "/data/" + dataset_name + "/masks/"
        if weight_map_name != None:
            self.maps_dir = cwd + "/data/" + dataset_name + "/" + weight_map_name + "/"
        else:
            self.maps_dir = None

        dataset_metadata = pd.read_csv(cwd + "/data/" + dataset_name + "/dataset_metadata.csv")

        mean = dataset_metadata["mean"].tolist()
        std = dataset_metadata["std"].tolist()

        self.norm = v2.Normalize(mean=mean, std=std)

        self.preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # scale=True: 0-255 to 0-1
        ])

        if augmentation:
            if dataset_name == "SNEMI3D" or dataset_name == "mass_road":
                self.transform = v2.Compose([
                    v2.RandomCrop(size=(512, 512)),
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip()
                ])
            elif dataset_name == "GlaS":
                self.transform = v2.Compose([
                    v2.RandomCrop(size=(448, 576)),
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip()
                ])
            else:
                self.transform = v2.Compose([
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

        if self.maps_dir != None:
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
    dataset = Dataset("mass_road", [0], True, None)
    print(len(dataset))
    img, msk, _, _ = dataset[0]
    print(img.shape)
    print(msk)
    print(torch.min(msk), torch.max(msk))
    print(torch.min(img), torch.max(img))
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(torch.permute(img, (1,2,0)).squeeze())
    axes[1].imshow(msk.squeeze())
    plt.show()
    