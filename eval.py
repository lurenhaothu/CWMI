import torch
from sklearn.model_selection import KFold
from model.dataset import Dataset
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
import model.att_unet as models

experiment_names = [
                    ]

def test_experiment(experiment_name):


    model = models.U_Net

    dataset_name = "GlaS"

    cwd = os.getcwd()

    result_folder = cwd

    curResultDir = result_folder + "/model/results/" + experiment_name + "/"

    test_metrics = [metrics.miou, metrics.vi, metrics.mdice, metrics.ari, metrics.hausdorff_distance]

    for fold in range(3):

        print("fold: " + str(fold))

        testIndexFile = curResultDir + 'Fold_' + str(fold) + '_test.csv'
        df = pd.read_csv(testIndexFile, header=None)
        test_list = df[0].tolist()[1:]

        test_dataset = Dataset(dataset_name, test_list, augmentation=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        unet = model(img_ch=3, output_ch=2)

        unet.cuda()

        #test
        best_state_dict = torch.load(curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
        unet.load_state_dict(best_state_dict)
        unet.eval()

        images_test = []
        masks_test = []
        preds_test = []
        preds_bin_test = []
        test_results = []
        for index, (image, mask, w_map, class_weight) in enumerate(test_dataloader):
            unet.eval()
            with torch.no_grad():
                pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
                # pred = torch.softmax(unet(image.cuda()), 1)
                # pred = pred.max(1)[1].data
                images_test.append(image.squeeze().numpy())
                masks_test.append(mask.squeeze().numpy())
                preds_test.append(pred.cpu().squeeze().numpy())
                preds_bin_test.append((preds_test[-1] > 0.5).astype(int))

        for test_metric in test_metrics:
            with ThreadPoolExecutor(max_workers=10) as executor:
                test_result = list(executor.map(test_metric, preds_bin_test, masks_test))
            test_results.append((test_metric.__name__, np.mean(test_result)))

        test_result_pd = {"Fold": [fold]}
        for metric_name, value in test_results:
            test_result_pd[metric_name] = [value]

        test_result_pd = pd.DataFrame(test_result_pd)

        if not os.path.exists(curResultDir + "_test_result_new.csv"):
            test_result_pd.to_csv(curResultDir + "_test_result_new.csv", index=False)
        else:
            test_result_pd.to_csv(curResultDir + "_test_result_new.csv", mode='a', header=False, index=False)

        print("---------------------------------------------------------")
        print(experiment_name)
        print("---------------------fold " + str(fold) + " finished---------------------")
        for metric_name, value in test_results:
            print(metric_name + ": " + str(value))
        print("---------------------------------------------------------")

for experiment_name in experiment_names:
    test_experiment(experiment_name)