import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz import _lovasz_hinge, _lovasz_softmax
from typing import Optional, Union


# binary loss
class BinaryDiceLoss(nn.Module):
    """
    binary dice loss
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        y_pred = F.logsigmoid(y_pred_logits).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth).clamp_min(self.eps)
        loss = 1. - dsc
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return loss.mean()


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

    def forward(self, y_pred_logits, y_true):
        diceloss = BinaryDiceLoss()
        dice = diceloss(y_pred_logits, y_true)
        bceloss = BinaryCrossEntropyLoss()
        bce = bceloss(y_pred_logits, y_true)
        return bce + dice


class MCC_Loss(nn.Module):
    """
    Compute Matthews Correlation Coefficient Loss for image segmentation task. It only supports binary mode.
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc


class BinaryLovaszLoss(nn.Module):
    def __init__(self, per_image: bool = False, ignore_index: Optional[Union[int, float]] = None):
        super(BinaryLovaszLoss).__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


# mutil loss

class MutilCrossEntropyLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        loss = F.cross_entropy(y_pred_logits.float(), y_true.long(), weight=self.alpha)
        return loss


class MutilFocalLoss(nn.Module):
    """
    """

    def __init__(self, alpha, gamma=2, torch=True):
        super(MutilFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.torch = torch

    def forward(self, y_pred_logits, y_true):
        if torch:
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)
            logpt = CE_loss(y_pred_logits.float(), y_true.long())
            pt = torch.exp(-logpt)
            loss = (((1 - pt) ** self.gamma) * logpt).mean()
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

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        y_pred = y_pred_logits.log_softmax(dim=1).exp()
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        smooth = 1.e-5
        eps = 1e-7
        assert y_pred.size() == y_true.size()
        intersection = torch.sum(y_true * y_pred, dim=(0, 2))
        denominator = torch.sum(y_true + y_pred, dim=(0, 2))
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
        loss = - gen_dice_coef
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return (loss * self.alpha).mean()


class LovaszLoss(nn.Module):
    """
    mutil LovaszLoss
    """

    def __init__(self, per_image=False, ignore=None):
        super(LovaszLoss, self).__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)
