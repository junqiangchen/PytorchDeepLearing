import torch
import torch.nn as nn


class LUConv2d(nn.Module):
    def __init__(self, nchan, prob=0.5):
        super(LUConv2d, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, nchan)
        self.drop = nn.Dropout2d(p=prob, inplace=True)

    def forward(self, x):
        out = self.relu1(self.drop(self.bn1(self.conv1(x))))
        return out


def _make_nConv2d(nchan, depth, prob=0.5):
    layers = []
    for _ in range(depth):
        layers.append(LUConv2d(nchan, prob=prob))
    return nn.Sequential(*layers)


class InputTransition2d(nn.Module):
    def __init__(self, inChans, outChans, prob=0.5):
        super(InputTransition2d, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inChans, outChans, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
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


class DownTransition2d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.5):
        super(DownTransition2d, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
        self.ops = _make_nConv2d(outChans, nConvs, prob)

    def forward(self, x):
        down = self.relu1(self.drop(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = torch.add(out, down)
        return out


class UpTransition2d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.5):
        super(UpTransition2d, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.GroupNorm(8, outChans)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
        self.ops = _make_nConv2d(outChans, nConvs, prob)
        self.conv = nn.Conv2d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx):
        out = self.relu(self.drop(self.bn(self.up_conv(x))))
        xcat = torch.cat((out, skipx), 1)
        xcat = self.relu(self.drop(self.bn(self.conv(xcat))))
        out = self.ops(xcat)
        # print(out.shape, xcat.shape)
        # assert 1>3
        out = torch.add(out, xcat)
        return out


class OutputTransition2d(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition2d, self).__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv = nn.Conv2d(inChans, outChans, kernel_size=1)

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


class VNet2d(nn.Module):
    """
    Vnet2d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass, init_features=16):
        super(VNet2d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass
        self.features = init_features

        self.in_tr = InputTransition2d(self.image_channel, self.features, 0.2)

        self.down_tr32 = DownTransition2d(self.features, self.features * 2, 2, 0.2)
        self.down_tr64 = DownTransition2d(self.features * 2, self.features * 4, 3, 0.2)
        self.down_tr128 = DownTransition2d(self.features * 4, self.features * 8, 3, 0.2)
        self.down_tr256 = DownTransition2d(self.features * 8, self.features * 16, 3, 0.2)

        self.up_tr256 = UpTransition2d(self.features * 16, self.features * 8, 3, 0.2)
        self.up_tr128 = UpTransition2d(self.features * 8, self.features * 4, 3, 0.2)
        self.up_tr64 = UpTransition2d(self.features * 4, self.features * 2, 2, 0.2)
        self.up_tr32 = UpTransition2d(self.features * 2, self.features, 1, 0.2)

        self.out_tr = OutputTransition2d(self.features, self.numclass)

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
