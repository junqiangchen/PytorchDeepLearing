import torch
import torch.nn as nn


class LUConv3d(nn.Module):
    def __init__(self, nchan, prob):
        super(LUConv3d, self).__init__()
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, nchan)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=prob, inplace=True)

    def forward(self, x):
        out = self.relu1(self.drop(self.bn1(self.conv1(x))))
        return out


def _make_nConv3d(nchan, depth, prob):
    layers = []
    for _ in range(depth):
        layers.append(LUConv3d(nchan, prob))
    return nn.Sequential(*layers)


class InputTransition3d(nn.Module):
    def __init__(self, inChans, outChans, prob=0.2):
        super(InputTransition3d, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.drop = nn.Dropout3d(p=prob, inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu1(self.drop(self.bn1(self.conv1(x))))
        # convert input to 16 channels
        x16 = self.relu1(self.drop(self.bn1(self.conv2(x))))
        # print("x16", x16.shape)
        # print("out:", out.shape)
        out = torch.add(out, x16)
        # assert 1>3
        return out


class DownTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.2):
        super(DownTransition3d, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.drop = nn.Dropout3d(p=prob, inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.ops = _make_nConv3d(outChans, nConvs, prob)

    def forward(self, x):
        down = self.relu1(self.drop(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = torch.add(out, down)
        return out


class UpTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.2):
        super(UpTransition3d, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.GroupNorm(8, outChans)
        self.drop = nn.Dropout3d(p=prob, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.ops = _make_nConv3d(outChans, nConvs, prob)
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx):
        out = self.relu(self.drop(self.bn(self.up_conv(x))))
        xcat = torch.cat((out, skipx), 1)
        xcat = self.relu(self.drop(self.bn(self.conv(xcat))))
        out = self.ops(xcat)
        # print(out.shape, xcat.shape)
        # assert 1>3
        out = torch.add(out, xcat)
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
    def __init__(self, image_channel, numclass, init_features=16):
        super(VNet3d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass
        self.features = init_features

        self.in_tr = InputTransition3d(self.image_channel, self.features, prob=0.2)

        self.down_tr32 = DownTransition3d(self.features, self.features * 2, 2, prob=0.2)
        self.down_tr64 = DownTransition3d(self.features * 2, self.features * 4, 3, prob=0.2)
        self.down_tr128 = DownTransition3d(self.features * 4, self.features * 8, 3, prob=0.2)
        self.down_tr256 = DownTransition3d(self.features * 8, self.features * 16, 3, prob=0.2)

        self.up_tr256 = UpTransition3d(self.features * 16, self.features * 8, 3, prob=0.2)
        self.up_tr128 = UpTransition3d(self.features * 8, self.features * 4, 3, prob=0.2)
        self.up_tr64 = UpTransition3d(self.features * 4, self.features * 2, 2, prob=0.2)
        self.up_tr32 = UpTransition3d(self.features * 2, self.features, 1, prob=0.2)

        self.out_tr = OutputTransition3d(self.feature, self.numclass)

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
