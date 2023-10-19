# DocuSegment-Pytorch: Binary Segmentation for Documents. 

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.0.1+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.11+-blue.svg?logo=python&style=for-the-badge" /></a>

### **Contents**
- [Quick Start](#quick-start)
- [Data](#data)
- [Usage](#usage)
    - [Training](#training)
    - [Predictions](#predictions)
    - [Pre-Trained Models](#pre-trained-models)
- [Performance](#performance)

- [Future Work](#future-work)
- [References](#references)

A custom implementation of the UNet architecture, designed and tuned specifically for the task of binary segmentation of documents and backgrounds.

Due to limited data availability, the model is trained on a modified synthetic dataset, more on this [here](#data). 

## **Quick Start**

**Note: Installing CUDA is strongly recommended if you plan on running this locally.**

1. Install Dependencies: 
```bash
pip install -r requirements.txt
```
2. Download Dataset 
```bash
gdown --fuzzy https://drive.google.com/file/d/1xtwNLGGpo9PNyUQrab26elyc10Hvkcgk/view?usp=sharing 
```
3. Move to src Directory
```bash
cd src
```
4. Run Training Procedure
```bash 
python3 train.py -tdp <PATH TO IMGS DIR> <PATH TO MASKS DIR> -vdp <PATH TO IMGS DIR> <PATH TO MASKS DIR> -sn <.pth filename> 
```

## **Usage**

You can download the dataset I used in the [Data](#data) section, or any other binary segmentation dataset. As long as the images are square. 

If you choose to use your own dataset, I recommend a data directory structure resembling the following: 
```
DocuSegment-Pytorch/
├── data/
│    └── <dataset name>/
│           ├─- train/
│           │   ├─- images/
│           │   └── masks/
│           └── valid/
│               ├─- images/
│               └── masks/
├── runs/
│  
├── samples/ 
│  
├── src/
│
⋮
⋮
└── README.md
```

### **Training**



### **Predictions**


### **Pre-Trained Models** 


## **Data**



## **Performance**


## **Future Work**

TODO

## **References**

TODO dataset reference 

TODO UNet Reference 