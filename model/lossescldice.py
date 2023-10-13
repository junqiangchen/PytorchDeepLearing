import torch
import torch.nn as nn


def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    shape = x.size().tolist()
    if shape == 4:
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
    if shape == 5:
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool3d(x * -1, (3, 3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
    return x


def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width) or (batch, channel,depth, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)


class Binary_Soft_cldice_loss(nn.Module):
    """
     calculate binary clDice loss
    """

    def __int__(self):
        super(Binary_Soft_cldice_loss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width),or (batch, channel, depth,height, width)

        '''
        cl_pred = soft_skeletonize(pred)
        target_skeleton = soft_skeletonize(target)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        cldsc = (2. * intersection.sum() + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth).clamp_min(self.eps)
        loss = 1. - cldsc
        return loss.mean()


class Mutil_Soft_cldice_loss(nn.Module):
    '''
    calculate mutil clDice loss
    '''

    def __int__(self, alpha):
        self.alpha = alpha
        self.bscldice = Binary_Soft_cldice_loss()

    def forward(self, input, target):
        '''
        inputs shape  (batch, channel, height, width),or (batch, channel, depth,height, width)
        '''
        Batchsize, Channel = input.shape[0], input.shape[1]
        y_true = target.long().contiguous().view(Batchsize, -1)
        y_true = torch.nn.functional.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        y_true = y_true.view(input.size())
        assert input.size() == y_true.size()
        dice = 0
        for channel in range(0, Channel):
            dice += self.bscldice(input[:, channel, ...], y_true[:, channel, ...]) * self.alpha[channel]
        return dice / Channel
