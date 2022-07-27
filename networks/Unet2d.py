import torch.nn as nn
from collections import OrderedDict
import torch


class UNet2d(nn.Module):
    """"
    2d Unet network
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet2d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = UNet2d._block(self.in_channels, self.features, name="enc1", prob=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(self.features, self.features * 2, name="enc2", prob=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(self.features * 2, self.features * 4, name="enc3", prob=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(self.features * 4, self.features * 8, name="enc4", prob=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet2d._block(self.features * 8, self.features * 16, name="bottleneck", prob=0.2)
        self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2d._block((self.features * 8) * 2, self.features * 8, name="dec4", prob=0.2)
        self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2d._block((self.features * 4) * 2, self.features * 4, name="dec3", prob=0.2)
        self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((self.features * 2) * 2, self.features * 2, name="dec2", prob=0.2)
        self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = UNet2d._block(self.features * 2, self.features, name="dec1", prob=0.2)
        self.conv = nn.Conv2d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out_logit = self.conv(dec1)

        if self.out_channels == 1:
            output = torch.sigmoid(out_logit)
        if self.out_channels > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(8, features)),
            (name + "drop1", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(8, features)),
            (name + "drop2", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
