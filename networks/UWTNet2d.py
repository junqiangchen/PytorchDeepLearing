import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class UWTNet2d(nn.Module):
    def __init__(self, in_channels, out_channels, wave_level=2, model_size='mid'):
        super(UWTNet2d, self).__init__()
        if model_size == 'small':
            num_channels = [32, 64, 128, 256, 512]
        elif model_size == 'mid':
            num_channels = [64, 128, 256, 512, 1024]
        elif model_size == 'large':
            num_channels = [128, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        self.in_conv = nn.Conv2d(in_channels, num_channels[0], kernel_size=1)

        self.encoder1 = self.conv_block(num_channels[0], num_channels[1], wave_level)
        self.encoder2 = self.conv_block(num_channels[1], num_channels[2], wave_level)
        self.encoder3 = self.conv_block(num_channels[2], num_channels[3], wave_level)
        self.encoder4 = self.conv_block(num_channels[3], num_channels[4], wave_level)

        self.middle = self.midconv_block(num_channels[4], num_channels[4])

        self.decoder4 = self.dconv_block(num_channels[4] * 2, num_channels[3], wave_level)
        self.decoder3 = self.dconv_block(num_channels[3] * 2, num_channels[2], wave_level)
        self.decoder2 = self.dconv_block(num_channels[2] * 2, num_channels[1], wave_level)
        self.decoder1 = self.dconv_block(num_channels[1] * 2, num_channels[0], wave_level)

        self.final_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)

    def midconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def conv_block(self, in_channels, out_channels, wave_level):
        return nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_levels=wave_level),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

    def dconv_block(self, in_channels, out_channels, wave_level):
        return nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_levels=wave_level),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.in_conv(x)
        enc1 = F.leaky_relu(self.encoder1(x))

        enc2 = F.leaky_relu(self.encoder2(enc1))

        enc3 = F.leaky_relu(self.encoder3(enc2))

        enc4 = F.leaky_relu(self.encoder4(enc3))

        middle = self.middle(enc4)

        dec4 = self.decoder4(torch.cat([middle, enc4], 1))
        dec4 = F.leaky_relu(F.interpolate(dec4, scale_factor=(2, 2), mode='bilinear'))

        dec3 = self.decoder3(torch.cat([dec4, enc3], 1))
        dec3 = F.leaky_relu(F.interpolate(dec3, scale_factor=(2, 2), mode='bilinear'))

        dec2 = self.decoder2(torch.cat([dec3, enc2], 1))
        dec2 = F.leaky_relu(F.interpolate(dec2, scale_factor=(2, 2), mode='bilinear'))

        dec1 = self.decoder1(torch.cat([dec2, enc1], 1))
        dec1 = F.leaky_relu(F.interpolate(dec1, scale_factor=(2, 2), mode='bilinear'))

        output = self.final_conv(dec1)

        return output


if __name__ == "__main__":
    net = UWTNet2d(1, 1, 2)
    in1 = torch.rand((1, 1, 512, 512))
    out = net(in1)
    print(out.size())
