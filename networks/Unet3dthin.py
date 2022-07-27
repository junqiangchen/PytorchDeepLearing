import torch.nn as nn
from collections import OrderedDict
import torch


class UNet3dthin(nn.Module):
    """
    Unet3dthin implement
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet3dthin, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = UNet3dthin._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3dthin._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3dthin._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3dthin._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = UNet3dthin._block(self.features * 8, self.features * 16, name="bottleneck")
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.decoder4 = UNet3dthin._block(self.features * 16, self.features * 8, name="dec4")
        self.decoder3 = UNet3dthin._block(self.features * 8, self.features * 4, name="dec3")
        self.decoder2 = UNet3dthin._block(self.features * 4, self.features * 2, name="dec2")
        self.decoder1 = UNet3dthin._block(self.features * 2, self.features, name="dec1")
        self.conv = nn.Conv3d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decoder4(bottleneck)
        dec4 = self.upsample(dec4)
        dec4 = torch.add(dec4, enc4)

        dec3 = self.decoder3(dec4)
        dec3 = self.upsample(dec3)
        dec3 = torch.add(dec3, enc3)

        dec2 = self.decoder2(dec3)
        dec2 = self.upsample(dec2)
        dec2 = torch.add(dec2, enc2)

        dec1 = self.decoder1(dec2)
        dec1 = self.upsample(dec1)
        dec1 = torch.add(dec1, enc1)

        out_logit = self.conv(dec1)

        if self.out_channels == 1:
            output = torch.sigmoid(out_logit)
        if self.out_channels > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output

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
            (name + "conv2", nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
        return block
