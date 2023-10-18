# predict.py

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from utils.datasets import *
from models import * 


def predict(model: torch.nn.Module, input: Image, device, scale_fact: float = 1.0):
    model.eval()

    img = preprocess(input, scale_fact, isMask=False)
    img = img.unsqueeze(0) 
    img = img.to(device)

    with torch.no_grad(): 
        pred = model(img).cpu()
        logits = F.sigmoid(pred).float()
        logits  = logits.argmax(dim=1)
    
    return logits[0].long().squeeze().numpy()


def mask_2_img(mask: np.array, mask_vals): 
    img = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)
    for label in mask_vals: 
                img[mask == label] = DocumentDataset.COLORMAP.get(label)
    return Image.fromarray(img)


#TODO add help descriptions
def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-w', '--weight_file', required=True)
    parser.add_argument('-ip', '--input_paths', nargs='+')
    parser.add_argument('-odir', '--output_dir', required=True)
    parser.add_argument('-sc', '--scale_fact', type=float, default=1.0)
    
    return parser.parse_args()


if __name__ == '__main__': 
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    match args.model:
        case 'fcn':
            model = FCN()
        case 'unet':
            model =  UNet(n_channels=3,n_classes=2,n_blocks=4,start=32) 
        case _:
            raise ValueError(f'Invalid model option \'{args.model}\'')
    
    state_dict = torch.load(os.path.join('models', 'saves', f'{args.weight_file}'), map_location=device)
    mask_vals = state_dict.pop('mask_values', [0, 1])

    model.load_state_dict(state_dict)
    model.to(device)
    
    input_paths = args.input_paths
    output_dir = args.output_dir

    for i,fpath in enumerate(input_paths): 
        fname = os.path.basename(fpath)
        img = load_img(fpath)
        pred_mask = predict(model, img, device, args.scale_fact)
        pred_img = mask_2_img(pred_mask, mask_vals)
        pred_img.save(os.path.join(output_dir,fname), format='png')



        
        



   
