import torch

class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        return torch.mean(- mask * torch.log(pred + 1e-7) - (1 - mask) * torch.log(1 - pred + 1e-7))

class BCE_withClassBalance(torch.nn.Module):
    def __init__(self):
        super(BCE_withClassBalance, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None): #20570595.0 84287005.0
        return torch.mean(- class_weight[:,1:2,:] * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))

class DiceLoss(torch.nn.Module):
    # arXiv:1606.04797v1
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        sum_p12 = torch.sum(torch.pow(mask, 2))
        sum_g12 = torch.sum(torch.pow(pred, 2))
        sum_p1g1 = torch.sum(mask * pred)
        sum_p02 = torch.sum(torch.pow(1 - mask, 2))
        sum_g02 = torch.sum(torch.pow(1 - pred, 2))
        sum_p0g0 = torch.sum((1 - mask) * (1 - pred))
        return 1 - 0.5 * (2 * sum_p1g1 / (sum_p12 + sum_g12 + 1e-7) +
                          2 * sum_p0g0 / (sum_p02 + sum_g02 + 1e-7))
    
class TverskyLoss(torch.nn.Module):
    # arXiv:1706.05721v1
    def __init__(self, alpha, beta):
        # alpha, beta test from: [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        sum_p1g1 = torch.sum(pred * mask)
        sum_p0g0 = torch.sum((1 - pred) * (1 - mask))
        sum_p1g0 = torch.sum(pred * (1 - mask))
        sum_p0g1 = torch.sum((1 - pred) * mask)
        return 1 - 0.5 * (sum_p1g1 / (sum_p1g1 + self.alpha * sum_p1g0 + self.beta * sum_p0g1 + 1e-7) + 
                          sum_p0g0 / (sum_p0g0 + self.alpha * sum_p0g1 + self.beta * sum_p1g0 + 1e-7))

class JaccardLoss(torch.nn.Module):
    # arXiv:2312.05391v1
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        sum_p1g1 = torch.sum(mask * pred)
        sum_p0g0 = torch.sum((1 - pred) * (1 - mask))
        sum_p1_g1_p1g1 = torch.sum(mask + pred - mask * pred)
        sum_p0_g0_p0g0 = torch.sum((1 - mask) + (1 - pred) - (1 - mask) * (1 - pred))
        B, C, H, W = mask.shape
        return 1 - 0.5 * (sum_p1g1 / (sum_p1_g1_p1g1 + 1e-7) + 
                          sum_p0g0 / (sum_p0_g0_p0g0 + 1e-7))
    
class FocalLoss(torch.nn.Module):
    # arXiv:1708.02002v2
    # test gamma with: 1.5, 2, 2.5, 3
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        return -torch.mean(class_weight[:,1:2,:] * mask * torch.pow(1 - pred, self.gamma) * torch.log(pred + 1e-7) +
                          class_weight[:,0:1,:] * (1 - mask) * torch.pow(pred, self.gamma) * torch.log(1 - pred + 1e-7))


class Unet_Weight_BCE(torch.nn.Module):
    def __init__(self):
        super(Unet_Weight_BCE, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        return torch.mean(- (w_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))
    
class Weight_Map_BCE(torch.nn.Module):
    def __init__(self):
        super(Weight_Map_BCE, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        if w_map != None:
            return torch.mean(- (w_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7) - (w_map + class_weight[:,0:1,:]) * (1 - mask) * torch.log(1 - pred + 1e-7))
        else:
            return torch.mean(- class_weight[:,1:2,:] * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))




