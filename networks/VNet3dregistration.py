import torch
import torch.nn as nn
import voxelmorph as vxm


class SpatialTransformergrid(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return new_locs


class LUConv3d(nn.Module):
    def __init__(self, nchan, prob):
        """
        InstanceNorm3d比GrouNorm推理速度快，且显存消耗低
        """
        super(LUConv3d, self).__init__()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn = nn.InstanceNorm3d(nchan, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout3d(p=prob, inplace=True)

    def forward(self, x):
        out = self.relu(self.drop(self.bn(self.conv(x))))
        return out


def _make_nConv3d(nchan, depth, prob):
    layers = []
    for _ in range(depth):
        layers.append(LUConv3d(nchan, prob))
    return nn.Sequential(*layers)


class InputTransition3d(nn.Module):
    def __init__(self, inChans, outChans, prob=0.2):
        super(InputTransition3d, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn = nn.InstanceNorm3d(outChans, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout3d(p=prob, inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu(self.drop(self.bn(self.conv1(x))))
        # convert input to 16 channels
        x16 = self.relu(self.drop(self.bn(self.conv2(x))))
        # print("x16", x16.shape)
        # print("out:", out.shape)
        out = torch.add(out, x16)
        # assert 1>3
        return out


class DownTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.2):
        super(DownTransition3d, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.InstanceNorm3d(outChans, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout3d(p=prob, inplace=True)
        self.ops = _make_nConv3d(outChans, nConvs, prob)

    def forward(self, x):
        down = self.relu(self.drop(self.bn(self.down_conv(x))))
        out = self.ops(down)
        out = torch.add(out, down)
        return out


class UpTransition3d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.2):
        super(UpTransition3d, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.InstanceNorm3d(outChans, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout3d(p=prob, inplace=True)
        self.ops = _make_nConv3d(outChans, nConvs, prob)
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx):
        out = self.relu(self.drop(self.bn(self.up_conv(x))))
        xcat = torch.cat((out, skipx), 1)
        xcat = self.relu(self.drop(self.bn(self.conv(xcat))))
        out = self.ops(xcat)
        # print(out.shape, xcat.shape)
        # assert 1>3
        out = torch.add(out, xcat)
        return out


class OutputTransition3d(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition3d, self).__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x):
        # print(x.shape) # 1, 16, 64, 128, 128
        # assert 1>3
        # convolve 16 down to 2 channels
        out_logit = self.conv(x)
        return out_logit


class VNet3dRegistration(nn.Module):
    """
    VNet3d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass, vol_size, init_features=16):
        super(VNet3dRegistration, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass
        self.features = init_features

        self.in_tr = InputTransition3d(self.image_channel, self.features, prob=0.2)

        self.down_tr32 = DownTransition3d(self.features, self.features * 2, 2, prob=0.2)
        self.down_tr64 = DownTransition3d(self.features * 2, self.features * 4, 2, prob=0.2)
        self.down_tr128 = DownTransition3d(self.features * 4, self.features * 8, 2, prob=0.2)
        self.down_tr256 = DownTransition3d(self.features * 8, self.features * 16, 2, prob=0.2)

        self.up_tr256 = UpTransition3d(self.features * 16, self.features * 8, 2, prob=0.2)
        self.up_tr128 = UpTransition3d(self.features * 8, self.features * 4, 2, prob=0.2)
        self.up_tr64 = UpTransition3d(self.features * 4, self.features * 2, 2, prob=0.2)
        self.up_tr32 = UpTransition3d(self.features * 2, self.features, 1, prob=0.2)

        self.out_tr = OutputTransition3d(self.features, self.numclass)
        # build transformer layer
        self.spatial_transformer_image = vxm.torch.layers.SpatialTransformer(vol_size)
        self.spatial_transformer_label = vxm.torch.layers.SpatialTransformer(vol_size, mode="nearest")

    def forward(self, input_moving_image, input_fixed_image, input_moving_label, input_fixed_label):
        # print("x.shape:", x.shape)
        x = torch.cat((input_moving_image, input_fixed_image), 1)
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
        # extract ddf
        ddf = self.out_tr(out)
        # warp the moving image with the transformer using network-predicted ddf
        moved_image = self.spatial_transformer_image(input_moving_image, ddf)
        moved_label = self.spatial_transformer_label(input_moving_label, ddf)
        # print("out:", out.shape)
        return moved_image, moved_label, ddf
