%cd /content/CWMI
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
from datetime import datetime
from data.map_gen_ABW import WeightMapLoss
from model.clDice.clDice import soft_dice_cldice
from model.rmi.rmi import RMILoss
from model.CWMI_loss.CWMI_loss import CWMI_loss
from plot_result import plot_result
from model.skea_topo.skea_topo_loss import Skea_topo_loss
import model.att_unet as models

lossFuncs = [(CWMI_loss(complex=True, spN = 4, spK=12, beta=1, lamb=0.9, mag=1), "")]

model_classes = [models.U_Net]

# dataset_names = ["DRIVE", "GlaS", "mass_road", "SNEMI3D"]
dataset_names = ["GlaS"]

def experiment(dataset_name, model_class, lossFunc, note=''):

    val_metric = metrics.miou

    test_metrics = [metrics.miou, metrics.vi, metrics.mdice, metrics.ari, metrics.hausdorff_distance]

    batch_size = 10

    epoch_num = 50

    gradient_plt = False

    expriment_name = model_class.__name__ + "_" + dataset_name + "_" + lossFunc.__class__.__name__ + "_" + note + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    cwd = os.getcwd()
    curResultDir = cwd + "/results/" + expriment_name + "/"
    os.makedirs(curResultDir, exist_ok=True)

    load_pretrain = False
    pretrain_dir_name = "SNEMI3D_BCE_CLassW_val_miou_btchSize_10_2025-01-14-18-43-31"

    pretrain_dir = cwd + "/results/" + pretrain_dir_name + "/"

    numFile = len(os.listdir(cwd + "/data/" + dataset_name + "/masks/"))
    fileList = [i for i in range(numFile)]

    fold_num = 3
    kf = KFold(n_splits=fold_num, shuffle=True)

    for fold, (train_and_val_list, test_list) in enumerate(kf.split(fileList)):

        print("fold: " + str(fold))

        train_size = int(0.8 * len(train_and_val_list))
        val_size = len(train_and_val_list) - train_size

        train_list = random.sample(train_and_val_list.tolist(), train_size)
        val_list = [i for i in train_and_val_list if i not in train_list]

        df = pd.DataFrame({'TrainNumbers': train_list})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_train.csv', index=False)
        df = pd.DataFrame({'ValNumbers': val_list})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_val.csv', index=False)
        df = pd.DataFrame({'TestNumbers': test_list.tolist()})
        df.to_csv(curResultDir + 'Fold_' + str(fold) + '_test.csv', index=False)

        if lossFunc.__class__.__name__ == "Unet_Weight_BCE":
            weight_map_name = "unet_maps"
        elif lossFunc.__class__.__name__ == "WeightMapLoss":
            weight_map_name = "ABW_maps"
        elif lossFunc.__class__.__name__ == "Skea_topo_loss":
            weight_map_name = "skea_topo_maps"
        else:
            weight_map_name = None

        train_dataset = Dataset(dataset_name, train_list, augmentation=True, weight_map_name=weight_map_name)
        val_dataset = Dataset(dataset_name, val_list, augmentation=False, weight_map_name=weight_map_name)
        test_dataset = Dataset(dataset_name, test_list, augmentation=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if dataset_name == "SNEMI3D":
            model = model_class(img_ch=1, output_ch=2)
        else: # all other datasets are RGB
            model = model_class(img_ch=3, output_ch=2)

        if load_pretrain:
            prev_dict = torch.load(pretrain_dir + "fold_" + str(fold) + "_best_model_state.pth")
            model.load_state_dict(prev_dict)

        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        val_res_list = []
        curmin_val_res = None

        for epoch in range(epoch_num):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("epoch: " + str(epoch))

            t1 = time.time()

            model.train()
            for i, (image, mask, w_map, class_weight) in enumerate(train_dataloader):
                image = image.cuda()
                mask = mask.cuda()
                w_map = w_map.cuda()
                pred = torch.softmax(model(image), 1)[:, 1:2, :, :]
                # print(pred.shape, mask.shape, w_map.shape)
                loss = lossFunc(mask, pred, w_map, class_weight.cuda(), epoch)
                if i == 0:
                  print(loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            scheduler.step()

            t2 = time.time()

            print("train time: ", t2 - t1)

            # torch.save(unet.state_dict(), "epoch_" + str(i) + ".pth")

            # validation
            images = []
            masks = []
            preds = []
            preds_bin = []
            for index, (image, mask, w_map, class_weight) in enumerate(val_dataloader):
                model.eval()
                with torch.no_grad():
                    pred = torch.softmax(model(image.cuda()), 1)[:, 1:2, :, :]
                    images.append(image.squeeze().numpy())
                    masks.append(mask.squeeze().numpy())
                    preds.append(pred.cpu().squeeze().numpy())
                    preds_bin.append((preds[-1] > 0.5).astype(int))

                # TODO: calculate loss grad on pixels

                if gradient_plt and epoch % 10 == 9 and index == 0:
                    plot_figures = []
                    plot_figures.append(image.squeeze().numpy())
                    plot_figures.append(mask.squeeze().numpy())
                    plot_figures.append(pred.cpu().squeeze().numpy())
                    plot_figures.append((pred.cpu().squeeze().numpy() > 0.5).astype(int))
                    dif = mask.squeeze().numpy() - (pred.cpu().squeeze().numpy() > 0.5).astype(int)
                    plot_figures.append(dif)

                    pred.requires_grad = True
                    l = lossFunc(mask.cuda(), pred, w_map.cuda(), class_weight.cuda())
                    grad = torch.autograd.grad(l, pred)
                    plot_figures.append(grad[0].cpu().squeeze().numpy())

                    fig_path = curResultDir + "fold_" + str(fold) + "_epoch_" + str(epoch) + "_val_sample.png"
                    plot_result(plot_figures, fig_path, size=plot_figures[-1].shape)


            with ThreadPoolExecutor(max_workers=10) as executor:
                val_res_temp = list(executor.map(val_metric, preds_bin, masks))
            val_res = np.mean(val_res_temp)

            t3 = time.time()
            print("val time: ", t3 - t2)

            val_res_list.append(val_res)
            result = pd.DataFrame({
                "Epoch": [epoch],
                val_metric.__name__: [val_res],
            })
            if not os.path.exists(curResultDir + "fold_" + str(fold) + "_val_result.csv"):
                result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", index=False)
            else:
                result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", mode='a', header=False, index=False)

            if curmin_val_res == None or val_res < curmin_val_res:
                curmin_val_res = val_res
                torch.save(model.state_dict(), curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
                print("save best model")

            print("epoch: " + str(epoch) + " " + val_metric.__name__ + ": " + str(val_res))

        #test
        best_state_dict = torch.load(curResultDir + "fold_" + str(fold) + "_best_model_state.pth", weights_only=True)
        model.load_state_dict(best_state_dict)
        model.eval()

        images_test = []
        masks_test = []
        preds_test = []
        preds_bin_test = []
        test_results = []
        for index, (image, mask, _, _) in enumerate(test_dataloader):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(image.cuda()), 1)[:, 1:2, :, :]
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
        torch.cuda.empty_cache()

for  model_class in model_classes:
    for lossFunc, note in lossFuncs:
        for dataset_name in dataset_names:
            experiment(dataset_name, model_class, lossFunc, note)