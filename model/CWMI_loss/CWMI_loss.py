import torch
import torchvision

from model.CWMI_loss.ComplexSteerablePyramid import ComplexSteerablePyramid
from model.CWMI_loss.SSIM import SSIM
import model.loss as loss

from PIL import Image
import matplotlib.pyplot as plt
import math

_POS_ALPHA = 5e-4

class CWMI_loss(torch.nn.Module):
    def __init__(self, complex, spN = 4, spK=4, lamb=0.9, CW_method="MI", select = None):
        # CW_method: MI: mutual information; L1: L1 distance; L2: L2 distance; SSIM: Structure SIMilarity
        super(CWMI_loss, self).__init__()
        self.sp = ComplexSteerablePyramid(complex=complex, N=spN, K=spK)
        self.complex = complex
        self.lamb = lamb
        self.BCEW = loss.BCE_withClassBalance()
        self.CW_method = CW_method
        if self.CW_method == "SSIM":
            self.ssim = SSIM()
        self.select = select
        #self.block_complex = block_complex

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        if epoch == 0:
            return self.BCEW(mask, pred, None, class_weight)
        sp_mask = self.sp(mask)
        sp_pred = self.sp(pred)
        mi_output = []
        for i in range(self.sp.N):
            if self.CW_method == "MI":
                if self.complex:
                    mi_output.append(torch.mean(self.complex_mi(sp_mask[i + 1], sp_pred[i + 1])).real)
                else:
                    mi_output.append(torch.mean(self.real_mi(sp_mask[i + 1], sp_pred[i + 1])))
            elif self.CW_method == "L1":
                mi_output.append(torch.mean(torch.sum(torch.abs(sp_mask[i + 1] - sp_pred[i + 1]), dim=2)))
            elif self.CW_method == "L2":
                mi_output.append(torch.mean(torch.sqrt(torch.sum(torch.pow(torch.abs(sp_mask[i + 1] - sp_pred[i + 1]), 2), dim=2))))
            elif self.CW_method == "SSIM":
                if self.complex:
                    mi_output.append(self.ssim(torch.abs(sp_mask[i + 1]).squeeze(1), torch.abs(sp_pred[i + 1]).squeeze(1)))
                else:
                    mi_output.append(self.ssim(sp_mask[i + 1].squeeze(1), sp_pred[i + 1].squeeze(1)))

        loss = self.BCEW(mask, pred, None, class_weight) * self.lamb
        if self.select == None:
            for i in range(self.sp.N):
                loss += mi_output[i]
        else:
            loss += mi_output[self.select]
        return loss

    def real_mi(self, mask, pred):
        B, C, A, H, W = mask.shape # A: angle, number of orientations of the steerable pyramid
        mask_flat = mask.view(B, C, A, H * W).type(torch.cuda.DoubleTensor)
        mask_mean = torch.mean(mask_flat, dim=3, keepdim=True)
        mask_centered = mask_flat - mask_mean

        pred_flat = pred.view(B, C, A, H * W).type(torch.cuda.DoubleTensor)
        pred_mean = torch.mean(pred_flat, dim=3, keepdim=True)
        pred_centered = pred_flat - pred_mean

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 1, 3, 2)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 1, 3, 2)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 1, 3, 2)))

        diag_matrix = torch.eye(A)
        inv_cov_pred = torch.inverse(var_pred + diag_matrix.type_as(var_pred) * _POS_ALPHA)

        cond_cov_mask_pred = var_mask - torch.matmul(torch.matmul(cov_mask_pred, inv_cov_pred), torch.permute(cov_mask_pred, (0, 1, 3, 2)))

        chol = torch.linalg.cholesky(cond_cov_mask_pred)
        return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    def complex_mi(self, mask, pred):
        B, C, A, H, W = mask.shape # A: angle, number of orientations of the steerable pyramid
        mask_flat = mask.view(B, C, A, H * W)
        mask_flat = torch.cat((torch.cat((mask_flat.real, mask_flat.imag), dim=2), torch.cat((-mask_flat.imag, mask_flat.real), dim=2)), dim=3)
        mask_mean = torch.mean(mask_flat, dim=3, keepdim=True)
        mask_centered = mask_flat - mask_mean

        pred_flat = pred.view(B, C, A, H * W)
        pred_flat = torch.cat((torch.cat((pred_flat.real, pred_flat.imag), dim=2), torch.cat((-pred_flat.imag, pred_flat.real), dim=2)), dim=3)
        pred_mean = torch.mean(pred_flat, dim=3, keepdim=True)
        pred_centered = pred_flat - pred_mean

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 1, 3, 2)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 1, 3, 2)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 1, 3, 2)))

        diag_matrix = torch.eye(2*A)
        inv_cov_pred = torch.inverse(var_pred + diag_matrix.type_as(var_pred) * _POS_ALPHA)

        cond_cov_mask_pred = var_mask - torch.matmul(torch.matmul(cov_mask_pred, inv_cov_pred), torch.permute(cov_mask_pred, (0, 1, 3, 2)))

        chol = torch.linalg.cholesky(cond_cov_mask_pred)
        return torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

