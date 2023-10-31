# inference.py

import os, sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from utils.datasets import *
from models import * 


def predict(model: torch.nn.Module, input: Image, device: torch.device, scale_fact: float = 1.0) -> np.ndarray:

    """ Preprocess and performs a forward pass on a single input image.

    Args: 
        model (nn.Module): model
        input (Image): Input Image
        device (torch.device): device where model is located
        scale_fact (float): Scale factor to reduce/enlarge the input image by 

    Returns: 
        logits (np.ndarray): Model's predicted segmentation of the input 
    """

    model.eval()

    img = DocumentDataset.preprocess(input, DocumentDataset.COMMON, scale_fact, isMask=False)
    img = img.unsqueeze(0) 
    img = img.to(device)

    with torch.no_grad(): 
        pred = model(img).cpu()
        if model.n_classes > 2:
            logits = F.softmax(pred, dim=1).float()
        else:
            logits = F.sigmoid(pred).float()
        logits  = logits.argmax(dim=1)
    
    return logits[0].long().squeeze().numpy()


def mask_2_img(mask: np.ndarray, mask_vals: np.ndarray) -> Image:

    """ Converts a segmentation mask into an image
    
    Args: 
        mask (np.ndarray): input mask
        mask_vals (np.ndarray): vector containing the unique mask values

    Returns: 
        Image: output image
    """

    img = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)
    for label in mask_vals: 
                img[mask == label] = DocumentDataset.COLORMAP.get(label)
    return Image.fromarray(img)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weight_file', metavar='filename', required=True, help="The saved model's filename.")
    parser.add_argument('-inputs', '--input_paths', metavar='path', nargs='+', help="Paths to the input(s).")
    parser.add_argument('-odir', '--output_dir', metavar='dir', required=True, help="Path to Directory where predictions are saved.")
    parser.add_argument('-sc', '--scale_fact', metavar='s', type=float, default=1.0, help="Factor to reduce / increase the inputs by.")
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run model.')
    
    return parser.parse_args()


if __name__ == '__main__': 
    args = get_args()

    device = torch.device(args.device)

    try: 
        load = torch.load(os.path.join('models', 'saves', f'{args.weight_file}'), map_location=device)
        state_dict = load['state_dict']

        model = UNet(n_channels=3, 
                      n_classes=load['n_classes'], 
                      n_blocks=load['n_blocks'], 
                      start=load['start_channels']
                    ) 
        mask_vals = list(range(load['n_classes']))
        model.load_state_dict(state_dict)
        model.to(device)
    
    except FileNotFoundError:
        print("Error: The model file was not found.")
        sys.exit(1)
    
    except KeyError:
        print("Error: The loaded state_dict does not match the model architecture.")
        sys.exit(1)
        
    except Exception as e: 
        print(f"Unexpected error: {e}")
        sys.exit(1)

    else:

        input_paths = args.input_paths
        output_dir = args.output_dir

        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        for i,fpath in enumerate(input_paths): 
            if not os.path.exists(fpath): 
                print(f"Warning: File {fpath} not found.")

            fname = os.path.basename(fpath)
            img = load_img(fpath)
            pred_mask = predict(model, img, device, args.scale_fact)
            pred_img = mask_2_img(pred_mask, mask_vals)
            pred_img.save(os.path.join(output_dir, fname), format='png')

            



        
        



   
