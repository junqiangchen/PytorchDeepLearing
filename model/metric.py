import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F
import math
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology
from skimage.metrics import structural_similarity as compare_ssim


class Seg_Metirc3d():
    """

    计算基于重叠度和距离等九种分割常见评价指标
    """

    def __init__(self, real_mask, pred_mask, voxel_spacing):
        """
        :param real_mask: 金标准
        :param pred_mask: 预测结果
        :param voxel_spacing: 体数据的spacing
        """
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self._get_surface(real_mask)
        self.pred_mask_surface_pts = self._get_surface(pred_mask)

        self.real2pred_nn = self._get_real2pred_nn()
        self.pred2real_nn = self._get_pred2real_nn()

    # 下面三个是提取边界和计算最小距离的实用函数
    def _get_surface(self, mask):
        """
        提取array的表面点的真实坐标(以mm为单位)
        :param mask: ndarray
        :return: 提取array的表面点的真实坐标(以mm为单位)
        """
        # 卷积核采用的是三维18邻域
        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask
        surface_pts = surface.nonzero()
        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))
        # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
        # 读出来的数据spacing和shape不是对应的,所以需要反向
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def _get_pred2real_nn(self):
        """
        预测结果表面体素到金标准表面体素的最小距离
        :return: 预测结果表面体素到金标准表面体素的最小距离
        """
        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)
        return nn

    def _get_real2pred_nn(self):
        """
        金标准表面体素到预测结果表面体素的最小距离
        :return: 金标准表面体素到预测结果表面体素的最小距离
        """
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)
        return nn

    # 下面的六个指标是基于重叠度的
    def get_dice_coefficient(self):
        """
        dice系数 dice系数的分子
        :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()
        return 2 * intersection / union, 2 * intersection, union

    def get_jaccard_index(self):
        """
        iou value
        :return: 杰卡德系数
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()
        return intersection / union

    def get_VOE(self):
        """
        体素重叠误差 Volumetric Overlap Error
        :return: 体素重叠误差 Volumetric Overlap Error
        """
        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """
        体素相对误差 Relative Volume Difference
        :return: 体素相对误差 Relative Volume Difference
        """
        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        """
        欠分割率 False negative rate
        :return: 欠分割率 False negative rate
        """
        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):
        """
        过分割率 False positive rate
        :return: 过分割率 False positive rate
        """
        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union

    # 下面的三个指标是基于距离的
    def get_ASSD(self):
        """
        平均表面距离 Average Symmetric Surface Distance
        :return: 对称位置平均表面距离 Average Symmetric Surface Distance
        """
        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
               (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        """
        均方根 Root Mean Square symmetric Surface Distance
        :return: 对称位置表面距离的均方根 Root Mean Square symmetric Surface Distance
        """
        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):
        """
        最大表面距离 Maximum Symmetric Surface Distance
        :return: 对称位置的最大表面距离 Maximum Symmetric Surface Distance
        """
        return max(self.pred2real_nn.max(), self.real2pred_nn.max())


# segmeantaion metric
def dice_coeff(input: Tensor, target: Tensor):
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
    mask = y_true_nobk.sum((0, 2)) > 0
    gen_dice_coef *= mask.to(gen_dice_coef.dtype)
    return gen_dice_coef.sum() / torch.count_nonzero(mask)


def multiclass_iou_coeff(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert input.size() == target.size()
    union = 0
    # remove backgroud region
    for channel in range(1, input.shape[1]):
        union += iou_coeff(y_pred[:, channel, ...], y_true[:, channel, ...])
    return union / (input.shape[1] - 1)


def multiclass_iou_coeffv2(input: Tensor, target: Tensor):
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
    intersection = (y_pred_nobk * y_true_nobk)
    union = (intersection.sum(1) + smooth) / (
            y_pred_nobk.sum(1) + y_true_nobk.sum(1) - intersection.sum(1) + smooth).clamp_min(eps)
    mask = y_true_nobk.sum((0, 2)) > 0
    union *= mask.to(union.dtype)
    return union.sum() / torch.count_nonzero(mask)


# classification metric

def calc_accuracy(input: Tensor, target: Tensor):
    n = input.size(0)
    acc = torch.sum(input == target).sum().float() / n
    return acc


def calc_mse(input: Tensor, target: Tensor):
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    mse = (input - target) ** 2
    return torch.mean(mse)


def calc_nrmse(input: Tensor, target: Tensor):
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    mse = (input - target) ** 2
    mse = torch.mean(mse)
    mse = torch.sqrt(mse)
    eps = 1e-7
    nrmse = 0
    for batch in range(num):
        min = torch.min(target[batch])
        max = torch.max(target[batch])
        nrmse += mse / (max - min + eps)
    return nrmse / num


# image quality metric
def calc_psnr(input: Tensor, target: Tensor, mean: Tensor, std: Tensor):
    """Peak Signal to Noise Ratio
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)"""
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    eps = 1e-7
    psnr = 0
    for batch in range(num):
        mse = torch.mean((input[batch] * std[batch] - target[batch] * std[batch]) ** 2)
        max = torch.max(target[batch] * std[batch] + mean[batch])
        psnr += 10 * torch.log10(max ** 2 / mse + eps)
    return psnr / num


def calc_ssim(input: Tensor, target: Tensor, mean: Tensor, std: Tensor):
    """ssim"""
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    eps = 1e-7
    ssim = 0
    for batch in range(num):
        real_image = (target[batch] * std[batch] + mean[batch]) / (
                torch.max(target[batch] * std[batch] + mean[batch]) + eps)
        pred_image = (input[batch] * std[batch] + mean[batch]) / (torch.max(
            input[batch] * std[batch] + mean[batch]) + eps)
        ssim += compare_ssim(real_image.detach().cpu().squeeze().numpy().astype(np.float16),
                             pred_image.detach().cpu().squeeze().numpy().astype(np.float16))
    return ssim / num
