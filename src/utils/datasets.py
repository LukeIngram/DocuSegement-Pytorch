# datasets.py

import os
import numpy as np 
from PIL import Image
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset


SUPPORTED_EXTS = ['.png','.jpeg','.jpg','.ppm','.gif','.tiff','.bmp']
SUPPORTED_SIZE = (312,312)


def load_img(fname): 
    ext = os.path.splitext(fname)[1]
    if ext not in SUPPORTED_EXTS: 
        raise ValueError(f"Unsupported filetype {ext}. Must be " + ', '.join(SUPPORTED_EXTS))
    else: 
        return Image.open(fname)


class DocumentDataset(Dataset): 

    def __init__(self,img_dir,mask_dir,scaleFact=1.0):
        self.img_paths = [os.path.join(img_dir,f) for f in tqdm(os.listdir(img_dir),desc="Loading Images")]
        self.mask_paths = [os.path.join(mask_dir,f) for f in tqdm(os.listdir(mask_dir),desc="Loading Masks")]
        self.type = type
        self.scaleFact = scaleFact #TODO IMPLEMENT IMAGE SCALING
    
    def __len__(self): 
        return len(self.imgs_dir)
    
    def __getitem__(self, index): 
        img_path = self.img_paths[index]
        img = np.asarray(load_img(img_path))
        img = img/255.0
    

        mask_path = self.mask_paths[index]
        mask = load_img(mask_path)
        mask = np.zeros((*SUPPORTED_SIZE,2),dtype=np.float32)

        mask[:,:,0] = np.where(mask[:,:,0] == 0,1.0,0.0)
        mask[:,:,1] = np.where(mask[:,:,0] == 255,1.0,0.0)

        return {
            "image" : torch.from_numpy(img).permute(2,0,1),
            "mask" : torch.from_numpy(mask).permute(2,0,1)
        }
        
            

        

        

    
    