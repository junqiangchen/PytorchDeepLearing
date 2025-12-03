import torch
from torch import nn
import math
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


# https://github.com/SLDGroup/MK-UNet

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes is None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activation = act_layer(activation, inplace=True)

        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)

        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GroupedAttentionGate, self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                      activation='relu6'):
    """
    create a series of multi-kernel inverted residual blocks.
    """
    convs = []
    xx = MultiKernelInvertedResidualBlock(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel,
                                          add=add, kernel_sizes=kernel_sizes, activation=activation)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MultiKernelInvertedResidualBlock(out_c, out_c, 1, expansion_factor=expansion_factor,
                                                  dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
                                                  activation=activation)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv


# channels = [4,8,16,24,32] for MK_UNet-T2d
# channels = [8,16,32,48,80] for MK_UNet-S2d
# channels = [16,32,64,96,160] for MK_UNet2d
# channels = [32,64,128,192,320] for MK_UNet-M2d
# channels = [64,128,256,384,512] for MK_UNet-L2d

class MK_UNet_T2d(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, channels=[4, 8, 16, 24, 32], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out

        ### Stage 4
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        ### Bottleneck
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        ### Stage 4
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return [p4]  # [p4, p3, p2, p1]


class MK_UNet_S2d(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, channels=[8, 16, 32, 48, 80], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out

        ### Stage 4
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        ### Bottleneck
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        ### Stage 4
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return [p4]  # [p4, p3, p2, p1]


class MK_UNet2d(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out

        ### Stage 4
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        ### Bottleneck
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        ### Stage 4
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return [p4]  # [p4, p3, p2, p1]


if __name__ == "__main__":
    net = MK_UNet2d(5)
    in1 = torch.rand((1, 1, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
