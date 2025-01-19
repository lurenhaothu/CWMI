import torch
import torchvision
import math

class SPMap(torch.nn.Module):
    def __init__(self, sigma=100., kernel_size=9, beta=2):
        super(SPMap, self).__init__()
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.beta = beta

    def fft_upsample_2(self, image: torch.tensor):
        B, C, H, W = image.shape

        f = torch.fft.rfft2(image, dim=(-2, -1))
        B, C, FH, FW = f.shape

        newf = torch.zeros((B, C, 2 * H, W + 1)).to(torch.cfloat).cuda()
        newf[:,:,:H // 2, :FW] = f[:,:,:H//2,:]
        newf[:,:, -H//2:, :FW] = f[:,:,-H//2:,:]

        res = torch.fft.irfft2(newf, dim=(-2, -1))
        return self.gaussian(torch.abs(res))
    
    def fft_upsample_n(self, image, scale):
        for i in range(scale):
            image = self.fft_upsample_2(image)
        return image
    
    def forward(self, sp_image):
        B, C, H, W = sp_image[1].shape
        sp_N = len(sp_image) - 2
        res = torch.zeros((B, 1, H, W)).to(sp_image[1].dtype).cuda()
        for i in range(sp_N):
            i_level_feature = torch.abs(torch.sum(sp_image[i + 1], dim=1, keepdim=True))
            res += math.pow(self.beta, i) * self.fft_upsample_n(i_level_feature, i)
        return res