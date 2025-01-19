import torch
from sklearn.model_selection import KFold
from model.dataset import SNEMI3DDataset
from torch.utils.data import DataLoader
from model.unet import UNet
from PIL import Image
import numpy as np
import model.metrics as metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import model.loss as loss
import time
from concurrent.futures import ThreadPoolExecutor

expriment_name = "SNEMI3D_DiceLoss_btchSize_102025-01-08 21:31:22"

post_process = True

cwd = os.getcwd()
curResultDir = cwd + "/results/" + expriment_name + "/"



for fold in range(3):

    print("fold: " + str(fold))

    testIndexFile = curResultDir + 'Fold_' + str(fold) + '_test.csv'
    df = pd.read_csv(testIndexFile, header=None)
    test_list = df[0].tolist()[1:]

    test_dataset = SNEMI3DDataset(test_list, augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    unet = UNet()

    unet.cuda()

    #test
    best_state_dict = torch.load(curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
    unet.load_state_dict(best_state_dict)
    unet.eval()

    images_test = []
    masks_test = []
    preds_test = []
    preds_bin_test = []
    vis_test = []
    for index, (image, mask, _) in enumerate(test_dataloader):
        unet.eval()
        with torch.no_grad():
            pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
            images_test.append(image.squeeze().numpy())
            masks_test.append(mask.squeeze().numpy())
            preds_test.append(pred.cpu().squeeze().numpy())
            preds_bin_test.append((preds_test[-1] > 0.5).astype(int))

    if post_process:
        for i in range(len(preds_bin_test)):
            preds_bin_test[i] = metrics.post_process_output(preds_bin_test[i])
            masks_test[i] = metrics.post_process_label(masks_test[i])

    with ThreadPoolExecutor(max_workers=10) as executor:
        vis_test = list(executor.map(metrics.vi, preds_bin_test, masks_test))
    vi_test = np.mean(vis_test)

    test_result = pd.DataFrame({
        "Fold": [fold],
        "VI": [vi_test],
    })
    if not os.path.exists(curResultDir + "_test_result.csv"):
        test_result.to_csv(curResultDir + "_test_result.csv", index=False)
    else:
        test_result.to_csv(curResultDir + "_test_result.csv", mode='a', header=False, index=False)

    print("---------------------------------------------------------")
    print("-----------------------fold finished---------------------")
    print("fold: " + str(fold) + " VI: " + str(vi_test))
    print("---------------------------------------------------------")