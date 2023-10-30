# DocuSegment-Pytorch: UNet Based Document Segmentation in PyTorch. 

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.0.1+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.11+-blue.svg?logo=python&style=for-the-badge" /></a>

### **Contents**
- [Quick Start](#quick-start)
- [Usage](#usage)
    - [Training](#training)
    - [Predictions](#predictions)
- [Synthetic Dataset](#synthetic-dataset)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

A custom implementation of the UNet architecture, designed and tuned specifically for the task of binary segmentation of documents and backgrounds.

Due to limited data availability, the model is trained on a modified synthetic dataset, more on this [here](#data). 

## **Quick Start**

**Note: Installing CUDA is strongly recommended for local training.**

1. Install Dependencies: 
```bash
pip install -r requirements.txt
```
2. Download Dataset 
```bash
gdown --fuzzy https://drive.google.com/file/d/1Sej8ssQbp13Dlqdv9EI4SXaUMGLciacK/view?usp=sharing
```
3. Run Training Procedure
```bash 
python3 train.py -tdp <PATH TO IMGS DIR> <PATH TO MASKS DIR> -vdp <PATH TO IMGS DIR> <PATH TO MASKS DIR> -sn <.pth filename> 
```
Resulting weights and training details will be saved to the *runs* and *models/saves* directory. 

## **Usage**

For detailed on the synthetic dataset mentioned earlier, see the [Synthetic Dataset](#synthetic-dataset) section.

If you choose to use your own dataset, the following structure is recommended: 
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
└── README.md
```

You may need to adapt the *DocumentDataset* class in *utils/datasets.py* to suit your dataset. 

### **Training**

```console
> python -m train -h 
usage: train.py [-h] [-nb N] [-nc N] [-n N] [-lr eta] [-bs b] [-sc s] [-tdp path [path ...]] [-vdp path [path ...]] -sn filename [-vbo] [-udi]

options:
  -h, --help            show this help message and exit
  -nb N, --num_blocks N
                        "Number of down sampling & upsampling blocks featured in the UNet."
  -nc N, --num_start_channels N
                        "Number of channels after the first convolution block."
  -n N, --num_epochs N  "Number of epochs."
  -lr eta, --learning_rate eta
                        "Learning Rate."
  -bs b, --batch_size b
                        "Batch Size."
  -sc s, --scale_fact s
                        "factor to reduce / increase the inputs by."
  -tdp path [path ...], --train_data_paths path [path ...]
                        "Paths of training image and mask directories."
  -vdp path [path ...], --validation_data_paths path [path ...]
                        "Paths of validation image and mask directories."
  -sn filename, --save_name filename
                        "same of save file (.pth)."
  -vbo, --verbose       "Verbose Output."
  -udi, --use_dice_and_iou
                        "Add DICE and IoU score to loss during training."
```


### **Inference**
```console
> python3 -m predict -h
usage: predict.py [-h] -w filename [-ip path [path ...]] -odir dir [-sc s]

options:
  -h, --help            show this help message and exit
  -w filename, --weight_file filename
                        "The saved model's filename."
  -ip path [path ...], --input_paths path [path ...]
                        "Paths to the input(s)."
  -odir dir, --output_dir dir
                        "Path to Directory where predictions are saved."
  -sc s, --scale_fact s
                        "factor to reduce / increase the inputs by."
```



## **Synthetic Dataset**



## **Results**


| Model Name | Num Params | Supported Size | Validation DICE | Validation IOU | 
|----|----|----|----|----| 
| UNet_16 | 1.9M | 480 x 480 | 0.| 0.|
| UNet_32 | 7.7M | 480 x 480 | 0. | 0.| 
| UNet_64 | 31M | 480 x 480 | 0. | 0. | 







## **Future Work**

- Proper Logging instead of printing to stdout
- Config file based args instead of passing through cmd line.
- Checkpointing / Loading and training existing model saves


## **References**

TODO dataset reference 

TODO UNet Reference 