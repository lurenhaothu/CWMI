import torch
import torchvision
from model.MI.steerable import SteerablePyramid
from model.MI.SPMap import SPMap
from PIL import Image
import matplotlib.pyplot as plt


path = "C:/Users/Renhao Lu/Desktop/Skea_topo-main/Para/data/masks/000.png"

imgg = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path)))

imgg = imgg.reshape((1,1,1024,1024)).cuda()

# imgg = torch.cat((imgg, imgg), dim=0)

print("imgg shape: ", imgg.shape)

SP = SteerablePyramid(imgSize=1024, K=4, N=4)

sp_output = SP(imgg)

print("layer 0 shape", sp_output[0].shape)
print("layer 1 shape", sp_output[1].shape)
print("layer 1 device", sp_output[1].device)

sp_map = SPMap()

a = sp_map(sp_output)

#plt.imshow(a.cpu().squeeze().numpy())
#plt.show()

path = "C:/Users/Renhao Lu/Desktop/dwt/test.jpg"

imgg = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path)))

C, H, W = imgg.shape

cf = torch.zeros((C, H, W), dtype=torch.cfloat)

f = torch.fft.rfft2(imgg)

C, FH, FW = f.shape

cf[:, :, 0:FW] = f[:, :, 0:FW] * 2
# cf[:, 0, 0] = cf[:, 0, 0] / 2

im = torch.fft.ifft2(cf)

fig, axe = plt.subplots(2,2)
axe[0][0].imshow(im.real.squeeze().cpu().numpy())
axe[0][1].imshow(im.imag.squeeze().cpu().numpy())
axe[1][0].imshow(im.abs().squeeze().cpu().numpy())
axe[1][1].imshow(im.angle().squeeze().cpu().numpy())
plt.show()