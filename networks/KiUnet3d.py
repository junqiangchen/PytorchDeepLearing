"""
This code is referenced from https://github.com/jeya-maria-jose/KiU-Net-pytorch
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class KiUNet3dthin(nn.Module):
    """
    # 该模型是一种轻量kiuNet实现
    """

    def __init__(self, in_channels, out_channels, init_features):
        super(KiUNet3dthin, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = KiUNet3dthin._block(self.in_channels, self.features, 'encoder1')
        self.encoder2 = KiUNet3dthin._block(self.features, self.features * 2, 'encoder2')
        self.encoder3 = KiUNet3dthin._block(self.features * 2, self.features * 4, 'encoder3')
        self.encoder4 = KiUNet3dthin._block(self.features * 4, self.features * 8, 'encoder4')
        self.encoder5 = KiUNet3dthin._block(self.features * 8, self.features * 16, 'encoder5')
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.downsample = nn.Upsample(scale_factor=(0.5, 0.5, 0.5), mode='trilinear')
        self.decoder1 = KiUNet3dthin._block(self.features * 16, self.features * 8, 'decoder1')
        self.map1 = KiUNet3dthin._outputblock(self.features * 8, self.out_channels, 8, 'map1')
        self.decoder2 = KiUNet3dthin._block(self.features * 8, self.features * 4, 'decoder2')
        self.map2 = KiUNet3dthin._outputblock(self.features * 4, self.out_channels, 4, 'map2')
        self.decoder3 = KiUNet3dthin._block(self.features * 4, self.features * 2, 'decoder3')
        self.map3 = KiUNet3dthin._outputblock(self.features * 2, self.out_channels, 2, 'map3')
        self.decoder4 = KiUNet3dthin._block(self.features * 2, self.features, 'decoder4')
        self.decoder5 = KiUNet3dthin._block(self.features, self.out_channels, 'decoder5')
        self.map4 = KiUNet3dthin._outputblock(self.features, self.out_channels, 1, 'map4')

        self.kencoder1 = KiUNet3dthin._block(self.in_channels, self.features, 'kencoder1')
        self.kdecoder1 = KiUNet3dthin._block(self.features, self.out_channels, 'kdecoder1')

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        out = self.decoder1(enc5)
        out = self.upsample(out)
        out = torch.add(out, enc4)
        output1_logit = self.map1(out)

        out = self.decoder2(out)
        out = self.upsample(out)
        out = torch.add(out, enc3)
        output2_logit = self.map2(out)

        out = self.decoder3(out)
        out = self.upsample(out)
        out = torch.add(out, enc2)
        output3_logit = self.map3(out)

        out = self.decoder4(out)
        out = self.upsample(out)
        out = torch.add(out, enc1)

        out1 = self.kencoder1(x)
        out1 = self.upsample(out1)
        out1 = self.kencoder2(out1)
        out1 = self.downsample(out1)

        out = self.decoder5(out)
        out = self.upsample(out)
        out = torch.add(out, out1)
        output4_logit = self.map4(out)

        if self.out_channels == 1:
            output1 = torch.sigmoid(output1_logit)
            output2 = torch.sigmoid(output2_logit)
            output3 = torch.sigmoid(output3_logit)
            output4 = torch.sigmoid(output4_logit)
        if self.out_channels > 1:
            output1 = torch.softmax(output1_logit, dim=1)
            output2 = torch.softmax(output2_logit, dim=1)
            output3 = torch.softmax(output3_logit, dim=1)
            output4 = torch.softmax(output4_logit, dim=1)
        return output1_logit, output2_logit, output3_logit, output4_logit, output1, output2, output3, output4

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))
        return block

    @staticmethod
    def _outputblock(in_channels, features, scale_factor, name):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=1,
                padding=1,
                bias=False, ),),
            (name + "up1", nn.Upsample(scale_factor=(scale_factor, scale_factor, scale_factor), mode='trilinear')),
        ]))
        return block


if __name__ == "__main__":
    net = KiUNet3dthin(1, 1, 16)
    in1 = torch.rand((1, 1, 64, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
