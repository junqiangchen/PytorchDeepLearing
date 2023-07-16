import torch
import torch.nn as nn


class LUConv3d(nn.Module):
    def __init__(self, nchan):
        super(LUConv3d, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv3d(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv3d(nchan))
    return nn.Sequential(*layers)


class InputTransition3d(nn.Module):
    def __init__(self, inChans, outChans):
        super(InputTransition3d, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu1(self.bn1(self.conv1(x)))
        # convert input to 16 channels
        x16 = self.relu1(self.bn1(self.conv2(x)))
        # print("x16", x16.shape)
        # print("out:", out.shape)
        out = torch.add(out, x16)
        # assert 1>3
        return out


class DownTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs):
        super(DownTransition3d, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(8, outChans)
        self.relu1 = nn.ReLU(inplace=True)
        self.ops = _make_nConv3d(outChans, nConvs)
        self.do1 = nn.Dropout3d(p=prob, inplace=True)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = torch.add(out, down)
        return out


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # suppose x is your feature map with size N*C*D*H*W
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # now x is of size N*C
        return x


class ResNet3d(nn.Module):
    """
    ResNet3d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass):
        super(ResNet3d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass

        self.in_tr = InputTransition3d(self.image_channel, 16)

        self.down_tr32 = DownTransition3d(16, 32, 2)
        self.down_tr64 = DownTransition3d(32, 64, 3)
        self.down_tr128 = DownTransition3d(64, 128, 3)
        self.down_tr256 = DownTransition3d(128, 256, 3)

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
