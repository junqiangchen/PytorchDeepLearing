from torch import Tensor
import torch


# segmeantaion metric
def dice_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    return dice


def iou_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    union = (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) - intersection.sum(1) + smooth)
    union = union.sum() / num
    return union


def multiclass_dice_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    dice = 0
    # remove backgroud region
    for channel in range(1, input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...])
    return dice / (input.shape[1] - 1)


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
    acc = torch.sum(input == target).sum().item() / n
    return acc
