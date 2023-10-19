# train.py

import os
import argparse
import configparser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm 
import matplotlib.pyplot as plt
from datetime import datetime

from models import *
from utils.datasets import DocumentDataset
from loss import dice_loss, IoU_loss
from evaluate import evaluate


def train(
        model: nn.Module, 
        device: str, 
        train_data_paths: tuple,            
        validation_data_paths: tuple, 
        save_name: str,
        epochs: int = 15, 
        batch_size: int = 8,
        learning_rate: float = 3e-4, 
        scale_fact: float = 1.0,
        verbose: bool = False,
        use_dice_iou: bool = True, 
    ):

    training_summary = []

    # Create & Load Datasets
    train = DocumentDataset(*train_data_paths, scale_fact)    
    val = DocumentDataset(*validation_data_paths, scale_fact)

    args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train, shuffle=True, **args)
    val_loader = DataLoader(val, shuffle=False, drop_last=True, **args)

    if verbose: 
        print(f'''
            Training with config: 
              Model:            {model}
              Device:           {device}
              SaveFile:         {save_name}.pth
              Epochs:           {epochs}
              Batch Size:       {batch_size}
              Training Size:    {len(train)}
              Validation Size:  {len(val)}
              Learning Rate:    {learning_rate}
              Scale Factor:     {scale_fact}
        ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs+1): 
        sampleCnt = 0
        training_loss = 0
        training_dice = 0
        training_iou = 0 

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} : training Loss 0") as pbar: 
            model.train()
            optimizer.zero_grad()
       
            for batch in train_loader: 
                inputs, truth = batch['image'], batch['mask'] 

                # Forward Pass
                inputs = inputs.to(device) 
                truth = truth.to(device, dtype=torch.long)

                pred = model(inputs)

                # Compute Loss 
                loss = criterion(pred.squeeze(1), truth.float())
                dice = dice_loss(F.sigmoid(pred).float(), truth, multiclass = (model.n_classes > 2))
                training_dice += inputs.shape[0] * (dice.item())

                iou = IoU_loss(F.sigmoid(pred).float(), truth, multiclass = (model.n_classes > 2))
                training_iou += inputs.shape[0] * (iou.item())

                # add iou & dice score to BCE Loss 
                if use_dice_iou: 
                    loss += (1.-dice) + (1.-iou)

                training_loss += inputs.shape[0] * loss.item()
                sampleCnt += inputs.shape[0]

                # Step & backward pass
                optimizer.zero_grad(set_to_none=True) 
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_description(f"Epoch {epoch}/{epochs} : training Loss {round(training_loss/sampleCnt,6)}")
        
        # Generate Summary
        epoch_summary = {}
        epoch_summary['loss'] = training_loss / sampleCnt
        epoch_summary['dice'] = training_dice / sampleCnt
        epoch_summary['iou'] = training_iou / sampleCnt

        eval_summary = evaluate(model, val_loader, device, epoch, epochs, criterion, use_dice_iou)
        scheduler.step(eval_summary['dice'] + eval_summary['iou'])

        if verbose: 
            print(f'''
                Epoch {epoch} Summary: 
                  Training Loss:                {epoch_summary.get('loss')}
                  Training Dice Coeff:          {epoch_summary.get('dice')}
                  Training IoU Score:           {epoch_summary.get('iou')}
                  Validation Loss:              {eval_summary.get('loss')}
                  Validation Dice Coeff:        {eval_summary.get('dice')}
                  Validation IoU Score:         {eval_summary.get('iou')}
            ''')

        training_summary.append({'training' : epoch_summary, 'validation' : eval_summary})

    # Save state dict & important attributes to saves dir 
    if verbose: 
        print(f"Saving Model Weights to {os.path.join('models', 'saves', f'{save_name}')}.pth")

    try:
        package = { 
            'state_dict': model.state_dict(), 
            'n_blocks': model.n_blocks,
            'n_classes': model.n_classes,
            'start_channels': model.start
        }
        torch.save(package,os.path.join('models', 'saves', f"{save_name}.pth"))

    except Exception as e: 
        print("Error saving model:", e)

    return training_summary


# Plots loss, dice, and iou scores & displays to user
def plot_summary(summary,save_name):
    n_plots = len(summary[0]['training'])
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 10))

    summary_agg = {
        category: { 
            metric: [item[category][metric] for item in summary]
            for metric in summary[0]['training'].keys() 
        }
        for category in summary[0].keys()
    }

    for idx,metric in enumerate(summary_agg['training']): 
        axes[idx].plot(summary_agg['training'][metric], marker='o', label="Training")
        axes[idx].plot(summary_agg['validation'][metric], marker='o', label='Validation')
        axes[idx].set_title(f"Training Epoch vs {metric}")
        axes[idx].set_xlabel('epoch')
        axes[idx].set_ylabel(f'{metric}')
        axes[idx].legend()

    fig.suptitle(f"Training Run: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}. Model Save: {save_name}")
    fig.tight_layout(pad=1.0)
    plt.show()
    


def get_args():
    config = configparser.ConfigParser()
    config.read(os.path.join('utils', 'cmdHelp.ini'))
    helpStrs = config['Train']

    parser = argparse.ArgumentParser() 
    parser.add_argument('-nb', '--num_blocks', metavar='N', type=int, default=4, help=helpStrs['num_blocks'])
    parser.add_argument('-nc', '--num_start_channels', metavar='N', type=int, default=32, help=helpStrs['num_start_channels'])
    parser.add_argument('-n', '--num_epochs', metavar='N', type=int, default=15, help=helpStrs['num_epochs'])
    parser.add_argument('-lr', '--learning_rate', metavar='eta', type=float, default=3e-4, help=helpStrs['learning_rate'])
    parser.add_argument('-bs', '--batch_size', metavar='b', type=int, default=8, help=helpStrs['batch_size'])
    parser.add_argument('-sc', '--scale_fact', metavar='s', type=float,default=1, help=helpStrs['scale_fact'])
    parser.add_argument('-tdp', '--train_data_paths', metavar='path', type=str, nargs='+', help=helpStrs['tdp'])
    parser.add_argument('-vdp', '--validation_data_paths', metavar='path', type=str, nargs='+', help=helpStrs['vdp'])
    parser.add_argument('-sn', '--save_name', metavar='filename', type=str, required=True, help=helpStrs['save_name'])
    parser.add_argument('-vbo', '--verbose', action='store_true', help=helpStrs['verbose'])
    parser.add_argument('-udi', '--use_dice_and_iou', action='store_true', help=helpStrs['udi'])
    
    return parser.parse_args()

# TODO REMOVE 
# SAMPLE CMD STRING: train.py -nb 1 -n 1 -bs 8 -tdp ..\data\document_dataset_resized\train\images\ ..\data\document_dataset_resized\train\masks\ -vdp ..\data\document_dataset_resized\valid\images\ ..\data\document_dataset_resized\valid\masks\ -sn debugging_1 -vbo -udi

if __name__ == '__main__': 
    args = get_args()
    
    for path in args.train_data_paths + args.validation_data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model =  UNet(n_channels=3, n_classes=2, n_blocks=args.num_blocks, start=args.num_start_channels) 
    model.to(device)

    if args.verbose: 
        print(summary(model, (3, 244, 244)))

    try:
        data = train(
            model=model,
            device=device,
            train_data_paths=args.train_data_paths,
            validation_data_paths=args.validation_data_paths,
            save_name=args.save_name,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            scale_fact=args.scale_fact,
            verbose=args.verbose,
            use_dice_iou=args.use_dice_and_iou,
        )

        plot_summary(data, args.save_name)

    except RuntimeError as e: 
        raise RuntimeError(f"Error during training: {e}")

        



    