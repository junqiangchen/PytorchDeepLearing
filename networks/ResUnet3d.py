import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResUNet3d(nn.Module):
    """
    Resnet+convdownsampleing+deepsuprivsion
    共9498260个可训练的参数, 接近九百五十万
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(ResUNet3d).__init__()

        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder_stage1 = ResUNet3d._block(self.in_channels, self.features, 'encoder_stage1')
        self.down_conv1 = ResUNet3d._downsamplingblock(self.features, self.features * 2, 'down_conv1')
        self.encoder_stage2 = ResUNet3d._block(self.features * 2, self.features * 2, 'encoder_stage2')
        self.down_conv2 = ResUNet3d._downsamplingblock(self.features * 2, self.features * 4, 'down_conv2')
        self.encoder_stage3 = ResUNet3d._block(self.features * 4, self.features * 4, 'encoder_stage3')
        self.down_conv3 = ResUNet3d._downsamplingblock(self.features * 4, self.features * 8, 'down_conv3')
        self.encoder_stage4 = ResUNet3d._block(self.features * 8, self.features * 8, 'encoder_stage4')
        self.down_conv4 = ResUNet3d._downsamplingblock(self.features * 8, self.features * 16, 'down_conv4')
        self.encoder_stage5 = ResUNet3d._block(self.features * 16, self.features * 16, 'encoder_stage5')
        self.up_conv1 = ResUNet3d._upsamplingblock(self.features * 16, self.features * 8, 'up_conv1')
        self.decoder_stage1 = ResUNet3d._block(self.features * 16, self.features * 8, 'decoder_stage1')
        self.map1 = ResUNet3d._outputblock(self.features * 8, self.out_channels, 8, 'map1')
        self.up_conv2 = ResUNet3d._upsamplingblock(self.features * 8, self.features * 4, 'up_conv2')
        self.decoder_stage2 = ResUNet3d._block(self.features * 8, self.features * 4, 'decoder_stage2')
        self.map2 = ResUNet3d._outputblock(self.features * 4, self.out_channels, 4, 'map2')
        self.up_conv3 = ResUNet3d._upsamplingblock(self.features * 4, self.features * 2, 'up_conv3')
        self.decoder_stage3 = ResUNet3d._block(self.features * 4, self.features * 2, 'decoder_stage3')
        self.map3 = ResUNet3d._outputblock(self.features * 2, self.out_channels, 2, 'map3')
        self.up_conv4 = ResUNet3d._upsamplingblock(self.features * 2, self.features, 'up_conv4')
        self.decoder_stage3 = ResUNet3d._block(self.features, self.features, 'decoder_stage4')
        self.map4 = ResUNet3d._outputblock(self.features, self.out_channels, 1, 'map4')

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3

        short_range4 = self.down_conv4(long_range4)
        long_range5 = self.encoder_stage5(short_range4) + short_range4

        short_range6 = self.up_conv1(long_range5)
        outputs = self.decoder_stage1(torch.cat([short_range6, long_range4], dim=1)) + short_range6
        output1_logit = self.map1(outputs)

        short_range7 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range7, long_range3], dim=1)) + short_range7
        output2_logit = self.map2(outputs)

        short_range8 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range8, long_range2], dim=1)) + short_range8
        output3_logit = self.map3(outputs)

        short_range9 = self.up_conv3(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range9, long_range1], dim=1)) + short_range9
        output4_logit = self.map4(outputs)

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

    @staticmethod
    def _downsamplingblock(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=2,
                padding=2,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))
        return block

    @staticmethod
    def _upsamplingblock(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=2,
                padding=2,
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
