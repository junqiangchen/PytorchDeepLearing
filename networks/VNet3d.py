import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features=16):
        super(ContBatchNorm3d, self).__init__(num_features=16)
        self.num_features = num_features

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv3d(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv3d, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(nchan)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv3d(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv3d(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition3d(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition3d, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.conv1(x)
        out = self.bn1(out)
        # convert input to 16 channels
        x16 = self.conv2(x)
        # print("x16", x16.shape)
        # print("out:", out.shape)
        out = self.relu1(torch.add(out, x16))
        # assert 1>3
        return out


class DownTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(DownTransition3d, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv3d(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)

        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition3d, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.GroupNorm(8, outChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv3d(outChans, nConvs, elu)

        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu(self.bn(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        xcat = self.relu(self.bn(self.conv(xcat)))
        out = self.ops(xcat)
        # print(out.shape, xcat.shape)
        # assert 1>3
        out = self.relu(torch.add(out, xcat))
        return out


class OutputTransition3d(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition3d, self).__init__()
        self.inChans = inChans
        self.outChans = outChans

        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x):
        # print(x.shape) # 1, 16, 64, 128, 128
        # assert 1>3
        # convolve 16 down to 2 channels
        out_logit = self.conv(x)

        if self.outChans == 1:
            output = torch.sigmoid(out_logit)
        if self.outChans > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output


class VNet3d(nn.Module):
    """
    VNet3d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass, elu=True):
        super(VNet3d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass

        self.in_tr = InputTransition3d(self.image_channel, 16, elu)

        self.down_tr32 = DownTransition3d(16, 32, 2, elu)
        self.down_tr64 = DownTransition3d(32, 64, 3, elu)
        self.down_tr128 = DownTransition3d(64, 128, 3, elu)
        self.down_tr256 = DownTransition3d(128, 256, 3, elu)

        self.up_tr256 = UpTransition3d(256, 128, 3, elu)
        self.up_tr128 = UpTransition3d(128, 64, 3, elu)
        self.up_tr64 = UpTransition3d(64, 32, 2, elu)
        self.up_tr32 = UpTransition3d(32, 16, 1, elu)

        self.out_tr = OutputTransition3d(16, self.numclass)

    def forward(self, x):
        # print("x.shape:", x.shape)
        out16 = self.in_tr(x)
        # print("out16.shape:", out16.shape) # 1, 16, 128, 128
        # assert 1>3
        out32 = self.down_tr32(out16)
        # print("out32.shape:", out32.shape) # 1, 32, 64, 64
        # assert 1>3
        out64 = self.down_tr64(out32)
        # print("out64.shape:", out64.shape) # 1, 64, 32, 32
        # assert 1>3
        out128 = self.down_tr128(out64)
        # print("out128.shape:", out128.shape) # 1, 128, 16, 16
        # assert 1>3
        out256 = self.down_tr256(out128)
        # print("out256.shape", out256.shape) # 1, 256, 8, 8
        # assert 1>3
        out = self.up_tr256(out256, out128)
        # print("out.shape:", out.shape)

        out = self.up_tr128(out, out64)
        # print("out:", out.shape)

        out = self.up_tr64(out, out32)
        # print("out:", out.shape)
        # assert 1>3
        out = self.up_tr32(out, out16)
        # print("last out:", out.shape)
        # assert 1>3
        out_logits, out = self.out_tr(out)
        # print("out:", out.shape)
        return out_logits, out
