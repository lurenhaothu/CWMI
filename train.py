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
from datetime import datetime
from data.map_gen_ABW import WeightMapLoss
from model.clDice.clDice import soft_dice_cldice
from model.rmi.rmi import RMILoss
from model.MI.SPMI import SPMILoss
from plot_result import plot_result

#lossFunc = RMILoss(num_classes=2, loss_weight_lambda=0.5, rmi_pool_size=3, rmi_pool_stride=3)
#lossFuncs = [(SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.5, ffl=False), SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.5, ffl=False)), 
#             (loss.BCE_withClassBalance(), loss.BCE_withClassBalance()),
#             (loss.DiceLoss(), loss.DiceLoss()),
#             (soft_dice_cldice(), soft_dice_cldice()),
#             (RMILoss(num_classes=2), RMILoss(num_classes=2)),
#             ]

lossFuncs = [(SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.1, mag=1, ffl=False), 
              SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.1, mag=1, ffl=False), "b_0.25_l_0.1_m_1"), 
             (SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.3, mag=1, ffl=False), 
              SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.3, mag=1, ffl=False), "b_0.25_l_0.3_m_1"), 
             (SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.5, mag=1, ffl=False), 
              SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.5, mag=1, ffl=False), "b_0.25_l_0.5_m_1"), 
             (SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.7, mag=1, ffl=False), 
              SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.7, mag=1, ffl=False), "b_0.25_l_0.7_m_1"), 
             (SPMILoss(imageSize=512, spN = 4, spK=4, beta=0.25, lamb=0.9, mag=1, ffl=False), 
              SPMILoss(imageSize=1024, spN = 4, spK=4, beta=0.25, lamb=0.9, mag=1, ffl=False), "b_0.25_l_0.9_m_1"), 
             ]

def experiment(lossFunc, lossFunc_val, note=''):

    val_metric = metrics.miou

    test_metrics = [metrics.miou, metrics.vi, metrics.mdice, metrics.ari] #,metrics.map_2018kdsb,  metrics.compute_bettis_own]

    batch_size = 10

    epoch_num = 50

    # 100 0.5053152359607173 0.19049978520827424 20570595.0 84287005.0

    expriment_name = "SNEMI3D_" + lossFunc.__class__.__name__ + "_" + note + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #expriment_name = "SNEMI3D_" + "SP_DIS_ALL" + "_val_" + val_name + "_btchSize_" + str(batch_size) + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    cwd = os.getcwd()
    curResultDir = cwd + "/results/" + expriment_name + "/"
    os.makedirs(curResultDir, exist_ok=True)

    load_prev = False
    pre_res_dir_name = "SNEMI3D_BCE_CLassW_val_miou_btchSize_10_2025-01-14-18-43-31"

    pre_res_dir = cwd + "/results/" + pre_res_dir_name + "/"

    numFile = 100
    fileList = [i for i in range(numFile)]

    fold_num = 3
    kf = KFold(n_splits=fold_num, shuffle=True)

    for fold, (train_and_val_list, test_list) in enumerate(kf.split(fileList)):

        print("fold: " + str(fold))

        train_size = int(0.8 * len(train_and_val_list))
        val_size = len(train_and_val_list) - train_size

        print(type(train_and_val_list))

        train_list = random.sample(train_and_val_list.tolist(), train_size)
        val_list = [i for i in train_and_val_list if i not in train_list]

        df = pd.DataFrame({'TrainNumbers': train_list})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_train.csv', index=False)
        df = pd.DataFrame({'ValNumbers': val_list})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_val.csv', index=False)
        df = pd.DataFrame({'TestNumbers': test_list.tolist()})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_test.csv', index=False)

        train_dataset = SNEMI3DDataset(train_list, augmentation=True, weight_map=False)
        val_dataset = SNEMI3DDataset(val_list, augmentation=False, weight_map=False)
        test_dataset = SNEMI3DDataset(test_list, augmentation=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        unet = UNet()

        if load_prev:
            prev_dict = torch.load(pre_res_dir + "fold_" + str(fold) + "_best_model_state.pth")
            unet.load_state_dict(prev_dict)

        unet.cuda()

        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        vi_record = []
        curmin_vi = None

        for epoch in range(epoch_num):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("epoch: " + str(epoch))

            t1 = time.time()

            unet.train()
            for i, (image, mask, w_map) in enumerate(train_dataloader):
                image = image.cuda()
                mask = mask.cuda()
                w_map = w_map.cuda()
                pred = torch.softmax(unet(image), 1)[:, 1:2, :, :]
                # print(pred.shape, mask.shape, w_map.shape)
                loss = lossFunc(mask, pred, w_map, epoch)
                if i == 0:
                  print(loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 5)
                optimizer.step()
            scheduler.step()

            t2 = time.time()

            print("train time: ", t2 - t1)

            # torch.save(unet.state_dict(), "epoch_" + str(i) + ".pth")

            # validation
            vi = 0.
            images = []
            masks = []
            preds = []
            preds_bin = []
            vis = []
            for index, (image, mask, w_map) in enumerate(val_dataloader):
                unet.eval()
                with torch.no_grad():
                    pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
                    images.append(image.squeeze().numpy())
                    masks.append(mask.squeeze().numpy())
                    preds.append(pred.cpu().squeeze().numpy())
                    preds_bin.append((preds[-1] > 0.5).astype(int))

                # TODO: calculate loss grad on pixels

                if epoch % 10 == 9 and index == 0:
                    plot_figures = []
                    plot_figures.append(image.squeeze().numpy())
                    plot_figures.append(mask.squeeze().numpy())
                    plot_figures.append(pred.cpu().squeeze().numpy())
                    plot_figures.append((pred.cpu().squeeze().numpy() > 0.5).astype(int))
                    dif = mask.squeeze().numpy() - (pred.cpu().squeeze().numpy() > 0.5).astype(int)
                    plot_figures.append(dif)

                    pred.requires_grad = True
                    l = lossFunc_val(mask.cuda(), pred, w_map.cuda())
                    grad = torch.autograd.grad(l, pred)
                    plot_figures.append(grad[0].cpu().squeeze().numpy())

                    fig_path = curResultDir + "fold_" + str(fold) + "_epoch_" + str(epoch) + "_val_sample.png"
                    plot_result(plot_figures, fig_path, size=1024)


            with ThreadPoolExecutor(max_workers=10) as executor:
                vis = list(executor.map(val_metric, preds_bin, masks))
            vi = np.mean(vis)

            t3 = time.time()
            print("val time: ", t3 - t2)

            vi_record.append(vi)
            result = pd.DataFrame({
                "Epoch": [epoch],
                val_metric.__name__: [vi],
            })
            if not os.path.exists(curResultDir + "fold_" + str(fold) + "_val_result.csv"):
                result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", index=False)
            else:
                result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", mode='a', header=False, index=False)

            if curmin_vi == None or vi < curmin_vi:
                curmin_vi = vi
                torch.save(unet.state_dict(), curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
                print("save best model")

            print("epoch: " + str(epoch) + " " + val_metric.__name__ + ": " + str(vi))

        #test
        best_state_dict = torch.load(curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
        unet.load_state_dict(best_state_dict)
        unet.eval()

        images_test = []
        masks_test = []
        preds_test = []
        preds_bin_test = []
        test_results = []
        for index, (image, mask, _) in enumerate(test_dataloader):
            unet.eval()
            with torch.no_grad():
                pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
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

        if not os.path.exists(curResultDir + "_test_result.csv"):
            test_result_pd.to_csv(curResultDir + "_test_result.csv", index=False)
        else:
            test_result_pd.to_csv(curResultDir + "_test_result.csv", mode='a', header=False, index=False)

        print("---------------------------------------------------------")
        print(expriment_name)
        print("---------------------fold " + str(fold) + " finished---------------------")
        for metric_name, value in test_results:
            print(metric_name + ": " + str(value))
        print("---------------------------------------------------------")

for lossFunc, lossFunc_val, note in lossFuncs:
    experiment(lossFunc, lossFunc_val, note)