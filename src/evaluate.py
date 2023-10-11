# evaluate.py

import torch
import torch.nn.functional as F 

from tqdm import tqdm

from loss import * 

@torch.inference_mode()
def evaluate(model, loader, device, epoch, epochs, criterion): 
    val_loss, val_dice, val_iou = 0, 0, 0 

    with tqdm(loader,desc=f'Epoch {epoch}/{epochs} : val Loss 0') as pbar: 
        model.eval()
        sampleCnt = 0 

        for batch in loader: 
            inputs, truth = batch['image'], batch['mask']

            inputs = inputs.to(device)
            truth = truth.to(device, dtype=torch.long) 

            pred = model(inputs)

            loss = criterion(pred.squeeze(1), truth.float()) 

            dice = dice_loss(
                    F.softmax(pred, dim=1).float(),
                    truth,
                    multiclass= (model.n_classes > 2)
                    )
            val_dice += inputs.shape[0] * (1.-dice.item())

            iou = IoU_loss(
                F.softmax(pred, dim=1).float(),
                truth,
                multiclass = (model.n_classes > 2)
                )
            val_iou += inputs.shape[0] * (1.-iou.item())

            val_loss += loss.item() * inputs.shape[0]
            sampleCnt += inputs.shape[0]

            pbar.update(1) 

            pbar.set_description(f'Epoch {epoch}/{epochs} : val Loss {round(val_loss/sampleCnt,6)}')

    summary = {}
    summary['loss'] = val_loss / sampleCnt
    summary['dice'] = val_dice / sampleCnt
    summary['iou'] = val_iou / sampleCnt 

    return summary

