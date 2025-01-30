# CWMI
The PyTorch implementation of the CWMI loss proposed in paper: ComplexWavelet Mutual Information Loss for Semantic Segmentations. <br>

## Abstract
Recent advancements in deep neural networks have significantly enhanced the performance of semantic segmentation. However, class imbalance and instance imbalance remain persistent challenges, particularly in biomedical image analysis, where smaller instances and thin boundaries are often overshadowed by larger structures. To address the multiscale nature of segmented objects, various models have incorporated mechanisms such as spatial attention and feature pyramid networks. Despite these advancements, most loss functions are still primarily pixel-wise, while regional and boundary-focused loss functions often incur high computational costs or are restricted to small-scale regions. To address this limitation, we propose complex wavelet mutual information (CWMI) loss, a novel loss function that leverages mutual information from subband images decomposed by a complex steerable pyramid. Unlike discrete wavelet transforms, the complex steerable pyramid captures features across multiple orientations, remains robust to small translations, and preserves structural similarity across scales. Moreover, mutual information is well-suited for capturing high-dimensional directional features and exhibits greater noise robustness compared to prior wavelet-based loss functions that rely on distance or angle metrics. Extensive experiments on diverse segmentation datasets demonstrate that CWMI loss achieves significant improvements in both pixel-wise accuracy and topological metrics compared to state-of-the-art methods, while introducing minimal computational overhead.

<p align = "center">
<img src="figures/Figure 1.PNG">
</p>

## Environment

    python 3.12.8
    pytorch 2.5.1+cu121

More depedencies are listed in requirements.txt

## Prepare datasets and running
- SNEMI3D can be download from https://zenodo.org/record/7142003, and it should be placed in ./data/snemi3d/.  <br>
- GlaS can be download from https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation, and it should be placed as ./data/GlaS/Warwick_QU_Dataset/.  <br>
- DRIVE can be download from https://www.kaggle.com/datasets/yurekanramasamy/drive-dataset, and it should be placed as ./data/DRIVE/Drive_source/.  <br>
- MASS ROAD can be download from https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset, and it should be placed as ./data/mass_road/mass_road_source/.  <br>

Usage Demo:

    # prepare datasets and calculate mean and std for each datasets
    python ./data/dataprepare.py

    # train
    run train.ipynb

    # eval 
    run eval.ipynb

CWMI doesn't need any other data preparation. However, if you would like to test other weight map based loss functions, please run below:

    # Unet weight map arXiv:1505.04597
    python ./data/map_gen_unet.py

    # ABW loss arXiv:1905.09226v2
    python ./data/map_gen_ABW.py

    # Skea_topo loss arXiv:2404.18539
    python ./data/skeleton_aware_loss_gen.py
    python ./data/skeleton_gen.py

