import math
from collections import Mapping, OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d

from timm.models.layers import DropPath, trunc_normal_
from timm.layers.helpers import to_2tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ATMOp(nn.Module):
    def __init__(
            self, in_chans, out_chans, stride: int = 1, padding: int = 0, dilation: int = 1,
            bias: bool = True, dimension: str = ''
    ):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        ATM along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f"{self.dimension} dimension not implemented")
        return deform_conv2d(
            input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation
        )

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'dimension={dimension}'
        s += ', in_chans={in_chans}'
        s += ', out_chans={out_chans}'
        s += ', stride={stride}'
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class ATMLayer(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim

        self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_h = ATMOp(dim, dim, dimension='h')
        self.atm_w = ATMOp(dim, dim, dimension='w')

        self.fusion = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f"offset shape not match, got {offset.shape}"
        w = self.atm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :]).permute(0, 2, 3, 1)
        h = self.atm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :]).permute(0, 2, 3, 1)
        c = self.atm_c(x)

        a = (w + h + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = w * a[0] + h * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'dim: {dim}'
        s += ')'
        return s.format(**self.__dict__)


class ActiveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 share_dim=1, downsample=None, new_offset=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.atm = ATMLayer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

        self.new_offset = new_offset
        self.share_dim = share_dim

        if new_offset:
            self.offset_layer = nn.Sequential(
                norm_layer(dim),
                nn.Linear(dim, dim * 2 // self.share_dim)
            )
        else:
            self.offset_layer = None

    def forward(self, x, offset=None):
        """
        :param x: [B, H, W, C]
        :param offset: [B, 2C, H, W]
        """
        if self.offset_layer and offset is None:
            offset = self.offset_layer(x).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1,
                                                                                            2)  # [B, H, W, 2C/S] -> [B, 2C, H, W]

        x = x + self.drop_path(self.atm(self.norm1(x), offset))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.downsample is not None:
            x_down = self.downsample(x)
            if self.offset_layer:
                return x_down, offset, x
            else:
                return x_down, x

        if self.offset_layer:
            return x, offset
        else:
            return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'new_offset: {new_offset}'
        s += ', share_dim: {share_dim}'
        s += ')'
        return s.format(**self.__dict__)


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PEG(nn.Module):
    """
    PEG
    from https://arxiv.org/abs/2102.10882
    """

    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PEG, self).__init__()
        # depth conv
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dim)
        self.stride = stride

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x_conv = x
        x_conv = x_conv.permute(0, 3, 1, 2)
        if self.stride == 1:
            x = self.proj(x_conv) + x_conv
        else:
            x = self.proj(x_conv)
        x = x.permute(0, 2, 3, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlaped patch embedding, implemeted with 2D conv
    """

    def __init__(self, patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = self.proj(x)
        return x


class ActiveMLP(nn.Module):
    """
    ActiveMLP
    https://arxiv.org/abs/2203.06108
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 4, 2],
            embed_dims=[64, 128, 320, 512],
            mlp_ratios=[4, 4, 4, 4],
            share_dims=[1, 1, 1, 1],  # how many channels share one offset
            drop_path_rate=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            intv=2,  # interval for generating new offset
            **kwargs
    ):

        super().__init__()

        self.depths = depths
        self.num_classes = num_classes
        self.intv = intv

        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        ii = 0
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            _block = nn.ModuleList([
                ActiveBlock(embed_dims[i],
                            mlp_ratio=mlp_ratios[i],
                            drop_path=dpr[ii + j],
                            share_dim=share_dims[i],
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            downsample=Downsample(embed_dims[i], embed_dims[i + 1]) if i < len(depths) - 1 and j ==
                                                                                       depths[i] - 1 else None,
                            new_offset=(j % self.intv == 0 and j != depths[i] - 1),
                            ) for j in range(depths[i])
            ])
            self.blocks.append(_block)
            ii += depths[i]

        # PEG for each resolution feature map
        self.pos_blocks = nn.ModuleList(
            [PEG(ed, ed) for ed in embed_dims]
        )

        # for neck
        for i in range(len(self.depths)):
            _layer = norm_layer(embed_dims[i])
            self.add_module(f'norm_{i}', _layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, ATMOp):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_blocks.' + n for n, p in self.pos_blocks.named_parameters()])

    def forward_blocks(self, x):
        outputs = []
        for i in range(len(self.depths)):
            for j, blk in enumerate(self.blocks[i]):
                if j % self.intv == 0 and j != len(self.blocks[i]) - 1:
                    # generate new offset
                    x = self.pos_blocks[i](x)
                    x, offset = blk(x)
                else:
                    # last stage or non-last layer per stage
                    if i == len(self.depths) - 1 or j != len(self.blocks[i]) - 1:
                        x = blk(x, offset)
                    else:  # last layer
                        x, y = blk(x, offset)
                        norm_layer = getattr(self, f'norm_{i}')
                        y_out = norm_layer(y)
                        outputs.append(y_out.permute(0, 3, 1, 2).contiguous())  # -> [B, C, H, W]

            if i == len(self.depths) - 1:
                norm_layer = getattr(self, f'norm_{i}')
                y_out = norm_layer(x)
                outputs.append(y_out.permute(0, 3, 1, 2).contiguous())  # -> [B, C, H, W]

        return outputs

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        x = self.forward_blocks(x)

        return x


class ActivexTiny(ActiveMLP):
    def __init__(self, **kwargs):
        depths = [3, 3, 3, 3]
        mlp_ratios = [4, 4, 4, 4]
        embed_dims = [64, 128, 256, 512]
        share_dims = [2, 4, 4, 8]
        super(ActivexTiny, self).__init__(
            depths=depths, embed_dims=embed_dims, mlp_ratios=mlp_ratios, share_dims=share_dims, intv=2, **kwargs
        )


class Amlp_Unet2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64):
        super(Amlp_Unet2d, self).__init__()
        features = init_features

        self.encoder = ActivexTiny(in_chans=in_channels)

        # model.default_cfg = default_cfg

        # self.encoder1 = UNet._block(in_channels, features, name="enc1")
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder2 = UNet._block(features, features * 2, name="enc2")
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        # self.upconv4 = nn.ConvTranspose2d(
        #     features * 16, features * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Amlp_Unet2d._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Amlp_Unet2d._block((features * 2) * 2, features * 2, name="dec2")
        # self.upconv1 = nn.ConvTranspose2d(
        #     features * 2, features, kernel_size=2, stride=2
        # )

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)

        self.decoder1 = Amlp_Unet2d._block(features * 2, features, name="dec1")

        self.outconv0 = nn.ConvTranspose2d(
            features, out_channels, kernel_size=8, stride=4, padding=2
        )
        # self.conv = nn.Conv2d(
        #     in_channels=features, out_channels=out_channels, kernel_size=1
        # )

    def forward(self, x):
        feature_list = self.encoder(x)
        # print(feature_list[0].shape)
        # print(feature_list[1].shape)
        # print(feature_list[2].shape)
        # print(feature_list[3].shape)

        # enc1 = self.encoder1(x)
        # enc2 = self.encoder2(self.pool1(enc1))
        # enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))
        # bottleneck = self.bottleneck(self.pool4(enc4))

        # dec4 = self.upconv4(feature_list[3])
        # dec4 = torch.cat((dec4, feature_list[2]), dim=1)
        # dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(feature_list[3])
        dec3 = torch.cat((dec3, feature_list[2]), dim=1)
        dec3 = self.decoder3(dec3)
        # print('dec3',dec3.shape)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, feature_list[1]), dim=1)
        dec2 = self.decoder2(dec2)
        # print('dec2',dec2.shape)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, feature_list[0]), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = torch.rand(dec1.shape, device=dec1.device) * 10 + dec1
        # print('dec1',dec1.shape)

        # return torch.sigmoid(self.conv(dec1))
        return self.outconv0(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == '__main__':
    x = torch.ones((2, 3, 512, 512)).cuda()
    # 注意h w 尺寸大小,magesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim
    net = Amlp_Unet2d(in_channels=3, out_channels=2, init_features=64).cuda()
    y = net(x)
    print(y.shape)
