# datasets.py

import os
import numpy as np 
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as vt


SUPPORTED_EXTS = ['.png','.jpeg','.jpg','.ppm','.gif','.tiff','.bmp','.JPG']


def load_img(fname: str) -> Image: 
    ext = os.path.splitext(fname)[1]
    if ext not in SUPPORTED_EXTS: 
        raise ValueError(f"Unsupported filetype {ext}. Must be " + ', '.join(SUPPORTED_EXTS))
    else: 
        return Image.open(fname)
    

class DocumentDataset(Dataset): 

    # Revise constants based on your needs 
    COLORMAP = {
        0 : (0, 0, 0), 
        1 : (255, 255, 255)
    }

    MEAN = (0.4611, 0.4359, 0.3905)
    STD = (0.2193, 0.2150, 0.2109) 

    TRAIN = vt.Compose([vt.ToTensor(), vt.RandomGrayscale(p=0.3), vt.Normalize(MEAN, STD)])
    COMMON = vt.Compose([vt.ToTensor(), vt.Normalize(MEAN, STD)])

    @staticmethod
    def preprocess(img: Image, transforms: vt.Compose, scale_fact: float = 1.0, isMask: bool = False) -> torch.Tensor:
        """
        TODO DOCSTRING 
        """ 
        h, w = img.size[:2]
        h, w = int(h * scale_fact), int(w * scale_fact)
        img = img.resize((h, w), Image.Resampling.LANCZOS)

        img = np.asarray(img, dtype=np.float32)

        if isMask: 
            out = np.zeros((*(h,w), 2), dtype=np.float32) 
            out[:, : , 0] = np.where(img[:, :, 0] == 0, 1.0, 0.0)
            out[:, : , 1] = np.where(img[:, :, 0] == 255, 1.0, 0.0)
            
            out = torch.from_numpy(out.transpose((2, 0, 1)).copy()).long().contiguous()

        else: 
            out = transforms(img)

        return out


    def __init__(self, img_dir: str, mask_dir: str, scale_fact: float = 1.0, isTrain: bool = False) -> None:
        self.img_paths = [os.path.join(img_dir,f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir,f) for f in os.listdir(mask_dir)]
        self.scale_fact = scale_fact 
        
        if isTrain: 
            self.transforms = self.TRAIN
        else: 
            self.transforms = self.COMMON
            

    
    def __len__(self) -> int: 
        return len(self.img_paths)
    
    
    def __getitem__(self, index) -> [torch.Tensor, torch.Tensor]: 
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = load_img(img_path)
        mask = load_img(mask_path)

        img = self.preprocess(img, self.transforms, self.scale_fact, isMask=False)
        mask = self.preprocess(mask, self.transforms, self.scale_fact, isMask=True)

        return {
            "image" : img,
            "mask" : mask
        }
        
            

        

        

    
    