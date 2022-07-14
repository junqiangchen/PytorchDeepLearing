import torch.nn as nn
import torch.nn.functional as F
import torch


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features=16):
        super(ContBatchNorm2d, self).__init__(num_features=16)
        self.num_features = num_features

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv2d(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv2d, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm2d(nchan)
        self.bn1 = nn.BatchNorm2d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv2d(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv2d(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition2d(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition2d, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inChans, outChans, kernel_size=1)
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


class DownTransition2d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(DownTransition2d, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv2d(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)

        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # suppose x is your feature map with size N*C*H*W
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # now x is of size N*C
        return x


class ResNet2d(nn.Module):
    """
    ResNet2d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass, elu=True):
        super(ResNet2d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass

        self.in_tr = InputTransition2d(self.image_channel, 16, elu)

        self.down_tr32 = DownTransition2d(16, 32, 2, elu)
        self.down_tr64 = DownTransition2d(32, 64, 3, elu)
        self.down_tr128 = DownTransition2d(64, 128, 3, elu)
        self.down_tr256 = DownTransition2d(128, 256, 3, elu)

        self.avg = GlobalAveragePooling()

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.numclass))

    def forward(self, x):
        # print("x.shape:", x.shape)
        x = self.in_tr(x)
        # print("x.shape:", x.shape) # 1, 16, 128, 128
        # assert 1>3
        x = self.down_tr32(x)
        # print("x.shape:", x.shape) # 1, 32, 64, 64
        # assert 1>3
        x = self.down_tr64(x)
        # print("x.shape:", x.shape) # 1, 64, 32, 32
        # assert 1>3
        x = self.down_tr128(x)
        # print("x.shape:", x.shape) # 1, 128, 16, 16
        # assert 1>3
        x = self.down_tr256(x)
        # print("x.shape", x.shape) # 1, 256, 8, 8
        x = self.avg(x)
        # print("x.shape", x.shape) # 1, 256
        x = self.fc_layers(x)
        # print("x.shape", x.shape) # 1, numclass
        return x
