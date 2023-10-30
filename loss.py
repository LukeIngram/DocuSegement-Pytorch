# loss.py
# Implementations for DICE and IoU loss used for model evaluation


import torch
from torch import Tensor


# Reference:
# Link:  https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_coeff(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor: 
    assert input.size() == target.size()

    sum_dim = (-1, -2, -3)

    # calculate Intersection
    inter = 2 * (input * target).sum(dim=sum_dim)

    # Calculate sum of input & target
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Calculate & return average DICE
    dice = (inter + eps) / (sets_sum + eps)
    return dice.mean()



def multiclass_dice_coeff(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), eps)



def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """
    TODO DOCSTRING 
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target)



def IoU_coeff(input: Tensor, target: Tensor, eps: float  = 1e-6) -> Tensor:
    assert input.size() == target.size()

    sum_dim = (-1, -2, -3)

    # Calculate Intersection
    inter = (input * target).sum(dim=sum_dim)

    # Calculate sum of input & target
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Calculate & return IoU score
    iou = (inter + eps) / (sets_sum - inter + eps)
    return iou.mean()



def multiclass_IoU_coeff(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor: 
    return IoU_coeff(input.flatten(0, 1), target.flatten(0, 1), eps) 


    
def IoU_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor: 
    """
    TODO DOCSTRING
    """
    fn = multiclass_IoU_coeff if multiclass else IoU_coeff 
    return 1 - fn(input, target) 

