import torch

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.upconv3 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.upconv4 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.finalconv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 2, 1)
        )

    def forward(self, image):
        e1 = self.conv1(image)
        e2 = self.conv2(self.maxpool(e1))
        e3 = self.conv3(self.maxpool(e2))
        e4 = self.conv4(self.maxpool(e3))
        e5 = self.conv5(self.maxpool(e4))
        e6 = self.conv6(torch.cat([e4, self.upconv1(e5)], 1))
        e7 = self.conv7(torch.cat([e3, self.upconv2(e6)], 1))
        e8 = self.conv8(torch.cat([e2, self.upconv3(e7)], 1))
        e9 = self.conv9(torch.cat([e1, self.upconv4(e8)], 1))
        output = self.finalconv(e9)
        return output

# test 
if __name__ == "__main__":
    testImage = torch.rand((4,1,512, 512))
    unet = UNet()
    pred = unet(testImage)
    print(pred.shape)