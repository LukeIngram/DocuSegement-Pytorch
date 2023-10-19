# predict.py

import os, sys
import argparse
import numpy as np
from PIL import Image
import configparser

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


def get_args(): 
    config = configparser.ConfigParser()
    config.read(os.path.join('utils', 'cmdHelp.ini'))
    helpStrs = config['Predict']

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file', metavar='filename', required=True, help=helpStrs['weight_file'])
    parser.add_argument('-ip', '--input_paths', metavar='path', nargs='+', help=helpStrs['input_paths'])
    parser.add_argument('-odir', '--output_dir', metavar='dir', required=True, help=helpStrs['output_dir'])
    parser.add_argument('-sc', '--scale_fact', metavar='s', type=float, default=1.0, help=helpStrs['scale_fact'])
    
    return parser.parse_args()


if __name__ == '__main__': 
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try: 
        load = torch.load(os.path.join('models', 'saves', f'{args.weight_file}'), map_location=device)
        state_dict = load['state_dict']

        model = UNet(n_channels=3, 
                      n_classes=load['n_classes'], 
                      n_blocks=load['n_blocks'], 
                      start=load['start_channels']
                    ) 
        mask_vals = state_dict.pop('mask_values', [0, 1])
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

            



        
        



   
