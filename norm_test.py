import torch
import torchvision
from model.MI.steerable import SteerablePyramid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

with torch.no_grad():

    mask_image_1 = torchvision.transforms.ToTensor()(Image.open('data/masks/004.png')).unsqueeze(0).cuda()

    sp = SteerablePyramid(imgSize=1024, N=4, K=4)

    mask = sp(mask_image_1)

    index = 1

    mask[index] = mask[index].squeeze(1)

    B, C, H, W = mask[index].shape
    print(B, C, H, W)
    mask_flat = mask[index].view(B, C, H * W).type(torch.cuda.DoubleTensor).squeeze(0)

    mask_test = torch.permute(mask_flat, (1, 0)).cpu().numpy()

    print(mask_test.shape)

    sample_size = 5000

    #sample = mask_test[np.random.choice(np.arange(H * W), size=sample_size), :]
    sample = mask_test
    
    import pingouin as pg

    #hz = pg.multivariate_normality(sample, alpha=.05)
    #print(hz)

    range_zero = 1e-20

    for c in range(C):
        s = sample[:, c]
        print(np.sum((s < range_zero) & (s > -range_zero)))
        
        # s = s[(s >= range_zero) | (s <= -range_zero)]

        #hist, bin_edges = np.histogram(sample[:, c])

        # Plot the histogram
        plt.hist(np.log10(np.abs(s) + range_zero), bins=100)
        plt.show()

        plt.hist(np.abs(s), bins=100)
        plt.show()

        s = np.log(np.abs(s) + range_zero)

        print(pg.normality(s))
        print(pg.normality(s, method='normaltest'))
        print(pg.normality(s, method='jarque_bera'))
        ax = pg.qqplot(s, dist='norm')
        plt.show()

        break

    '''
    mask_centered = mask_flat - torch.mean(mask_flat, dim=1, keepdim=True)

    mask_cov = torch.matmul(mask_centered, torch.permute(mask_centered, (1, 0)))

    diag_matrix = torch.eye(C).unsqueeze(0)
    _POS_ALPHA = 5e-4
    inv_mask_cov = torch.inverse(mask_cov + diag_matrix.type_as(mask_cov) * _POS_ALPHA)

    b1p = torch.mean(torch.matmul(torch.matmul(torch.permute(mask_centered, (1, 0)), inv_mask_cov), mask_centered))

    print(b1p)
    '''

