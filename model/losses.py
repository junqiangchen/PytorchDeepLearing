import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


# binary loss
class BinaryDiceLoss(nn.Module):
    """
    binary dice loss
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].float().contiguous().view(-1)
        y_true = y_true[:, 0].float().contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


class BinaryCrossEntropyLoss(nn.Module):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    pytorch推荐使用binary_cross_entropy_with_logits,
    将sigmoid层和binaray_cross_entropy合在一起计算比分开依次计算有更好的数值稳定性，这主要是运用了log-sum-exp技巧。
    """

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, y_pred_logits, y_true):
        assert y_pred_logits.size() == y_true.size()
        bce = F.binary_cross_entropy_with_logits(y_pred_logits.float(), y_true.float())
        return bce


class BinaryFocalLoss(nn.Module):
    """
    binary focal loss
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred_logits, y_true):
        assert y_pred_logits.size() == y_true.size()
        """
        https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
        """
        BCE_loss = F.binary_cross_entropy_with_logits(y_pred_logits.float(), y_true.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        """
        # p = torch.sigmoid(y_pred_logits)
        # p_t = p * y_true + (1 - p) * (1 - y_true)
        # loss = BCE_loss * ((1 - p_t) ** self.gamma)
        # if self.alpha >= 0:
        #     alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        #     focal_loss = alpha_t * loss

        return focal_loss.mean()


class BinaryCrossEntropyDiceLoss(nn.Module):
    """
    binary ce and dice
    """

    def __init__(self):
        super(BinaryCrossEntropyDiceLoss, self).__init__()

    def forward(self, y_pred, y_pred_logits, y_true):
        diceloss = BinaryDiceLoss()
        dice = diceloss(y_pred, y_true)
        bceloss = BinaryCrossEntropyLoss()
        bce = bceloss(y_pred_logits, y_true)
        return bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        input = y_pred.squeeze(1).float()
        target = y_true.squeeze(1).float()
        loss = lovasz_hinge(input, target, per_image=True)
        return loss


# mutil loss

class MutilCrossEntropyLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        loss = F.cross_entropy(y_pred.float(), y_true.float(), weight=self.alpha)
        return loss


class MutilFocalLoss(nn.Module):
    """
    """

    def __init__(self, alpha, gamma=2, torch=True):
        super(MutilFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.torch = torch

    def forward(self, y_pred, y_true):
        if torch:
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)
            logpt = CE_loss(y_pred.float(), y_true.float())
            pt = torch.exp(-logpt)
            loss = (((1 - pt) ** self.gamma) * logpt).mean()
        else:
            # not work
            # write version
            Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
            y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
            y_true = y_true.float().contiguous().view(Batchsize, Channel, -1)
            epsilon = 1.e-5
            # scale preds so that the class probas of each sample sum to 1
            y_pred = y_pred / torch.sum(y_pred, dim=1, keepdim=True)
            # manual computation of crossentropy
            y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
            celoss = - y_true * torch.log(y_pred)
            # manual computation of focal loss
            loss = torch.pow(1 - y_pred, self.gamma) * celoss
            loss = torch.sum(loss, dim=-1)
            loss = torch.mean(loss, dim=0)
            loss = torch.mean(self.alpha * loss)
        return loss


class MutilDiceLoss(nn.Module):
    """
        multi label dice loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        smooth = 1.e-5
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.float().contiguous().view(Batchsize, Channel, -1)
        intersection = y_true * y_pred
        intersection = intersection.sum(dim=-1)
        denominator = y_true + y_pred
        denominator = denominator.sum(dim=-1)
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).mean(dim=0)
        loss = - (gen_dice_coef * self.alpha).mean()
        return loss
