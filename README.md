# CWMI
The PyTorch implementation of the CWMI loss proposed in paper: **Complex Wavelet Mutual Information Loss: A Multi-Scale loss function for Semantic Segmentation**. <br>

---


## Abstract
Recent advancements in deep neural networks have significantly enhanced the performance of semantic segmentation. However, class imbalance and instance imbalance remain persistent challenges, particularly in biomedical image analysis, where smaller instances and thin boundaries are often overshadowed by larger structures. To address the multiscale nature of segmented objects, various models have incorporated mechanisms such as spatial attention and feature pyramid networks. Despite these advancements, most loss functions are still primarily pixel-wise, while regional and boundary-focused loss functions often incur high computational costs or are restricted to small-scale regions. To address this limitation, we propose complex wavelet mutual information (CWMI) loss, a novel loss function that leverages mutual information from subband images decomposed by a complex steerable pyramid. Unlike discrete wavelet transforms, the complex steerable pyramid captures features across multiple orientations, remains robust to small translations, and preserves structural similarity across scales. Moreover, mutual information is well-suited for capturing high-dimensional directional features and exhibits greater noise robustness compared to prior wavelet-based loss functions that rely on distance or angle metrics. Extensive experiments on diverse segmentation datasets demonstrate that CWMI loss achieves significant improvements in both pixel-wise accuracy and topological metrics compared to state-of-the-art methods, while introducing minimal computational overhead.

<p align = "center">
<img src="figures/Figure 1.PNG">
</p>

---


## **Environment**
Ensure you have the following dependencies installed:

```sh
Python 3.12.8
PyTorch 2.5.1+cu121
```

Additional dependencies are listed in [`requirements.txt`](./requirements.txt). Install them using:

```sh
pip install -r requirements.txt
```

---

## **Dataset Preparation**
Download the datasets and place them in the following directories:

| Dataset       | Download Link | Expected Directory |
|--------------|--------------|--------------------|
| **SNEMI3D**  | [Zenodo](https://zenodo.org/record/7142003) | `./data/snemi3d/` |
| **GlaS**     | [Kaggle](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation) | `./data/GlaS/Warwick_QU_Dataset/` |
| **DRIVE**    | [Kaggle](https://www.kaggle.com/datasets/yurekanramasamy/drive-dataset) | `./data/DRIVE/Drive_source/` |
| **MASS ROAD** | [Kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) | `./data/mass_road/mass_road_source/` |

---

## **Usage**
### **1. Prepare Datasets**
To preprocess datasets and calculate the mean and standard deviation for each dataset, run:

```sh
python ./data/dataprepare.py
```

### **2. Train the Model**
Run the training script:

```sh
run train.ipynb
```

### **3. Evaluate the Model**
Run the evaluation script:

```sh
run eval.ipynb
```

---

## **Additional Weight Map-Based Loss Functions**
CWMI does **not** require additional data preparation. However, if you wish to test other weight map-based loss functions, run the corresponding scripts:

- **U-Net weighted cross entropy (WCE)** ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597))
  ```sh
  python ./data/map_gen_unet.py
  ```
- **ABW loss** ([arXiv:1905.09226v2](https://arxiv.org/abs/1905.09226v2))
  ```sh
  python ./data/map_gen_ABW.py
  ```
- **Skea_topo loss** ([arXiv:2404.18539](https://arxiv.org/abs/2404.18539))
  ```sh
  python ./data/skeleton_aware_loss_gen.py
  python ./data/skeleton_gen.py
  ```

---


## Results
<p align = "center">
<img src="figures/Table 1.png">
</p>

### SNEMI3D
<p align = "center">
<img src="figures/Figure 3.PNG">
</p>

### GlaS
<p align = "center">
<img src="figures/Figure 4.PNG">
</p>

### DRIVE
<p align = "center">
<img src="figures/Figure 5.PNG">
</p>

### MASS ROAD
<p align = "center">
<img src="figures/Figure 6.PNG">
</p>

### Computational cost
<p align = "center">
<img src="figures/Table 4.png">
</p>
