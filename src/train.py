# train.py

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import dataloader,random_split

from tqdm import tqdm 

from models import UNet
from utils.datasets import DocumentDataset

def train(args):
    # Load Train & Validation Datasets 
    train = DocumentDataset(args[])


def get_args(): #TODO ADD HELP DESCRIPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb','--num_blocks',type=int,default=4)
    parser.add_argument('-n', '--num_epochs',type=int,default=20)
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3)
    parser.add_argument('-bs', '--batch_size',type=int,default=8)
    #parser.add_argument('-l','--load',type=str,default=None)
    parser.add_argument('-tdl','--train_data_location',type=str)
    parser.add_argument('-vdl','--valid_data_location',type=str,nargs='+')
    parser.add_argument('-rs','--random_seed',type=int,default=42)
    
    return parser.parse_args()

# SAMPLE CMD STRING: train.py -nb 1 -n 1 -bs 8 -tdl \\E:\\GitHub\\docUNET-Pytorch\\data\\document_dataset_resized\\train -vdl \\E:\\GitHub\\docUNET-Pytorch\\data\\document_dataset_resized\\train -rs 42

if __name__ == '__main__': 
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    model = UNet(n_channels=3,n_classes=2,n_blocks=args.num_blocks,start=32) 
    
    