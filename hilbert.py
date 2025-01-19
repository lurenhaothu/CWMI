import torch
import torchvision
from model.MI.SPMI import SPMILoss
#from steerable import SteerablePyramid
import model.loss as loss

from PIL import Image
import matplotlib.pyplot as plt
import math

from model.MI.SPMap import SPMap

import numpy as np

path1 = "C:/Users/Renhao Lu/Desktop/Skea_topo-main/Para/data/masks/000.png"
path2 = "C:/Users/Renhao Lu/Desktop/Skea_topo-main/Para/data/masks/001.png"

img1 = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path1))).unsqueeze(0).cuda()
img2 = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path1))).unsqueeze(0).cuda()

loss1 = SPMILoss(complex=True, spN=4, spK=12)
loss2 = SPMILoss(complex=False, spN=4, spK=12)

print(loss1(img1, img2, None, torch.reshape(torch.tensor([1,1]), (1,2,1)).cuda()))

print(loss2(img1, img2, None, torch.reshape(torch.tensor([1,1]), (1,2,1)).cuda())) 

