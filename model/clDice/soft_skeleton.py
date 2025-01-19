
import torch
import torch.nn.functional as F
#import matplotlib.pyplot as plt

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        #fig, axe = plt.subplots(1,3)
        #axe[0].imshow(img.squeeze())
        #axe[1].imshow(img1.squeeze())
        #axe[2].imshow(skel.squeeze())
        #plt.show()

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

            #fig, axe = plt.subplots(1,4)
            #axe[0].imshow(img.squeeze())
            #axe[1].imshow(img1.squeeze())
            #axe[2].imshow(delta.squeeze())
            #axe[3].imshow(skel.squeeze())
            #plt.show()

        return skel

    def forward(self, img):

        return self.soft_skel(img)


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms.v2 as v2
    mask = v2.ToImage()(Image.open('data/masks/000.png')).to(torch.float) / 255

    print(mask.shape)

    mask = mask.unsqueeze(0)
    
    s = SoftSkeletonize()

    sk = s(mask)
    
    fig, axe = plt.subplots(1,2)
    axe[0].imshow(mask.squeeze())
    axe[1].imshow(sk.squeeze())
    plt.show()
