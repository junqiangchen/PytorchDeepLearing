import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F


# segmeantaion metric
def dice_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    return dice


def iou_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    union = (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) - intersection.sum(1) + smooth)
    union = union.sum() / num
    return union


def multiclass_dice_coeff(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert y_pred.size() == y_true.size()
    dice = 0
    # remove backgroud region
    for channel in range(1, y_true.shape[1]):
        dice += dice_coeff(y_pred[:, channel, ...], y_true[:, channel, ...])
    return dice / (input.shape[1] - 1)


def multiclass_dice_coeffv2(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert y_pred.size() == y_true.size()
    smooth = 1e-5
    eps = 1e-7
    # remove backgroud region
    y_pred_nobk = y_pred[:, 1:Channel, ...]
    y_true_nobk = y_true[:, 1:Channel, ...]
    intersection = torch.sum(y_true_nobk * y_pred_nobk, dim=(0, 2))
    denominator = torch.sum(y_true_nobk + y_pred_nobk, dim=(0, 2))
    gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
    return gen_dice_coef.mean()


def multiclass_iou_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    union = 0
    # remove backgroud region
    for channel in range(1, input.shape[1]):
        union += iou_coeff(input[:, channel, ...], target[:, channel, ...])
    return union / (input.shape[1] - 1)


# classification metric

def calc_accuracy(input: Tensor, target: Tensor):
    n = input.size(0)
    acc = torch.sum(input == target).sum() / n
    return acc
