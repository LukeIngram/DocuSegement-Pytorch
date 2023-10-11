# predict.py

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from utils.datasets import *
from models import * 


#TODO
def predict(model: torch.nn.Module, input: Image, device, scale_fact: float = 1.0, out_threshhold: float = 0.5):
    model.eval()
    print(not np.asarray(input).transpose(2,0,1).any() )
    img = torch.from_numpy(preprocess(input, scale_fact, isMask=False))
    img = img.unsqueeze(0) 
    img = img.to(device)

    
    pred = model(img).cpu()
    print(pred.shape)
    print(pred)
    mask = pred.argmax(dim=1)
    print(not mask.any())
    print(mask.shape)
    return mask[0].squeeze().numpy()


def mask_2_img(mask: np.array, mask_vals): 
    print(not mask.any())
   # print(np.abs(mask[0]) > np.abs(mask[1]))
    print(mask)
    #mask = np.argmax(mask, axis=0)
    print(mask.shape)
    print(mask)

    print(not mask.any())
    img = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)
    mask = np.argmax(mask, axis=0)
    
    for i,v in enumerate(mask_vals): 
        img[mask == i] = v
    
    return Image.fromarray(img)

#TODO add help descriptions
def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-wf', '--weight_file', required=True)
    parser.add_argument('-ip', '--input_paths', nargs='+')
    parser.add_argument('-od', '--output_dir', required=True)
    parser.add_argument('-sc', '--scale_fact', type=float, default=1.0)
    
    return parser.parse_args()


if __name__ == '__main__': 
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    match args.model:
        case 'FCN':
            model = FCN()
        case 'UNet':
            model =  UNet(n_channels=3,n_classes=2,n_blocks=1,start=32) 
        case _:
            raise ValueError(f'Invalid model option \'{args.model}\'')
    
    state_dict = torch.load(os.path.join('models', 'saves', f'{args.weight_file}'), map_location=device)
    mask_vals = state_dict.pop('mask_values', [0, 1])
    print(type(mask_vals[0]))
    model.load_state_dict(state_dict)
    model.to(device)
    
    input_paths = args.input_paths
    output_dir = args.output_dir

    for i,fpath in enumerate(os.listdir("E:\\GitHub\\docUNET-Pytorch\\samples\\imgs\\")): 
        fname = os.path.basename(fpath) 

        img = load_img("E:\\GitHub\\docUNET-Pytorch\\samples\\imgs\\" +fpath)
        pred_mask = predict(model=model, input=img, device=device, scale_fact=args.scale_fact)
        pred_img = mask_2_img(pred_mask, mask_vals)

        pred_img.save(os.path.join(output_dir,fname), format='png')



        
        



   
