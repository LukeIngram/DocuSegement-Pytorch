# datasets.py

import os
import numpy as np 
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as vT


SUPPORTED_EXTS = ['.png','.jpeg','.jpg','.ppm','.gif','.tiff','.bmp','.JPG']


def transform():
    transforms = vT.Compose([vT.ToTensor()])
    return transforms


def load_img(fname): 
    ext = os.path.splitext(fname)[1]
    if ext not in SUPPORTED_EXTS: 
        raise ValueError(f"Unsupported filetype {ext}. Must be " + ', '.join(SUPPORTED_EXTS))
    else: 
        return Image.open(fname)

def preprocess(img, scale_fact: float = 1.0, isMask: bool = False, isTraining: bool = False): 
    h, w = img.size[:2]
    h, w = int(h * scale_fact), int(w * scale_fact)
    img.thumbnail((h, w), Image.Resampling.LANCZOS)

    img = np.asarray(img, dtype=np.float32)

    if isMask: 
        out = np.zeros((*(h,w), 2), dtype=np.float32) 
        out[:, : , 0] = np.where(img[:, :, 0] == 0, 1.0, 0.0)
        out[:, : , 1] = np.where(img[:, :, 0] == 255, 1.0, 0.0)
        
        out = torch.from_numpy(out.transpose((2, 0, 1)).copy()).long().contiguous()

        if not out.any(): 
            print("zero mask loaded")


    else: 
        out = transform()(img)
        out = out / 255.0

    return out


class DocumentDataset(Dataset): 

    COLORMAP = {
        0 : (0, 0, 0), 
        1 : (255, 255, 255)
    }

    def __init__(self, img_dir, mask_dir, scale_fact: float = 1.0):
        self.img_paths = [os.path.join(img_dir,f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir,f) for f in os.listdir(mask_dir)]
        self.scale_fact = scale_fact 
    
    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, index): 
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = load_img(img_path)
        mask = load_img(mask_path)

        img = preprocess(img, self.scale_fact, isMask=False)
        mask = preprocess(mask, self.scale_fact, isMask=True)

        return {
            "image" : img,
            "mask" : mask
        }
        
            

        

        

    
    