from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetUpBlock
from monai.networks.blocks.selfattention import SABlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.layers.factories import Pool

rearrange, _ = optional_import("einops", name="rearrange")
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["VSNet"]


class GDT(nn.Module):
    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


###################ours
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: type[LayerNorm] = nn.LayerNorm,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class SwinLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            depth: int,
            window_size: Sequence[int],
            drop_path: int = 0.0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: type[LayerNorm] = nn.LayerNorm,
            normalize: bool = True,
            downsample: nn.Module | None = None,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))
        normalize = normalize

    def proj_out(self, x, normalize):
        if normalize:
            x_shape = x.size()
            n, ch, d, h, w = x_shape
            x = rearrange(x, "n c d h w -> n d h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n d h w c -> n c d h w")
        return x

    def forward(self, x):
        # x = self.patch_embed(x)
        x_shape = x.size()
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x_out = rearrange(x, "b d h w c -> b c d h w")
        # x_out = self.proj_out(x, normalize)
        return x_out


class DepTran(nn.Module):

    def __init__(
            self, in_channels, out_channels
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.groupconv1 = nn.Conv3d(in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=True,
                                    groups=in_channels)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.groupconv1(out)
        x1, x2 = torch.chunk(out, 2, dim=1)
        x1 = self.gelu(x1)
        x2 = self.sigmoid(x2)
        out = x1 * x2
        out = out + x
        out = self.conv2(out)
        out = self.relu(out)
        return out


class Gate(nn.Module):

    def __init__(self, in_channels_up, in_channels_down, out_channels):
        super(Gate, self).__init__()
        self.w1 = nn.Sequential(
            nn.Conv3d(in_channels_up, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(out_channels)
        )
        self.w2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels_down, out_channels, kernel_size=2, stride=2, bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(out_channels)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        w1 = self.w1(x1)
        w2 = self.w2(x2)
        psi = self.relu(w1 + w2)
        psi = self.psi(psi)
        return x1 * psi


class CSA(nn.Module):
    def __init__(
            self,
            in_chans: int,
            # img_size=(96,96,96),
            img_size: int,
            dropout_rate: float = 0.0,
            save_attn: bool = False,

    ) -> None:
        super().__init__()
        # d, w, h = img_size[0]//16, image_size[1]//16, image_size[2]//16,
        d = img_size // 16
        h = img_size // 16
        w = img_size // 16
        self.norm = nn.LayerNorm([d, w, h])
        self.scale = (d * w * h) ** -0.5
        self.groupconv = nn.Conv3d(
            in_channels=in_chans,
            out_channels=in_chans * 3,
            kernel_size=1,
            groups=in_chans
        )
        self.q_rearrange = Rearrange("b q d h w-> b q (d h w)", d=d, h=h, w=w)
        self.k_rearrange = Rearrange("b c d h w-> b c (d h w)", d=d, h=h, w=w)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x):
        out = self.norm(x)
        out = self.groupconv(out)
        q, k, v = torch.chunk(out, 3, dim=1)
        q = self.q_rearrange(q)
        k = self.q_rearrange(k)
        att_mat = (torch.einsum("bqx,bcx->bqc", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.att_mat = att_mat.detach()
        att_mat = self.drop_weights(att_mat)
        x_out = torch.einsum("bqc,bcdhw->bqdhw", att_mat, v)
        x_out = self.drop_output(x_out)
        x_out = x + x_out
        return x_out


class SSA(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            img_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            save_attn: bool = False,
            dim_head: int | None = None,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        d = img_size // 16
        h = img_size // 16
        w = img_size // 16
        self.re1 = Rearrange("b c d h w-> b d h w c")
        self.norm = nn.LayerNorm(hidden_size)
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads

        self.transpose = Rearrange("b d h w c -> b (d h w) c")

        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b x (qkv l n) -> qkv b l x n", qkv=3, l=num_heads)
        self.out_rearrange1 = Rearrange("b l x n -> b x (l n)", l=num_heads, x=d * h * w)
        self.out_rearrange2 = Rearrange("b (d h w) c-> b d h w c", d=d, h=h, w=w)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head ** -0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

        self.re2 = Rearrange("b d h w c -> b c d h w")

    def forward(self, x):
        out = self.re1(x)
        out = self.norm(out)
        out = self.transpose(out)
        output = self.input_rearrange(self.qkv(out))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxn,blyn->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        out = torch.einsum("blxy,blyn->blxn", att_mat, v)
        out = self.out_rearrange1(out)
        out = self.out_proj(out)
        out = self.drop_output(out)
        out = self.out_rearrange2(out)
        out = self.re2(out)
        out = out + x
        return out


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            save_attn: bool = False,
            dim_head: int | None = None,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads

        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head ** -0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class Outlayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(Outlayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(out_channels)
        )
        if activation == "Sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "Softmax":
            self.act = nn.Softmax(dim=1)
        else:
            raise ValueError('Unproper Activation')

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class VSNet(nn.Module):
    def __init__(
            self,
            # img_size: Sequence[int] | int,
            in_channels: int,
            out_channels: int,
            depth: int = 2,
            img_size: int = 96,
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 12,
            norm_name: tuple | str = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = False,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
            use_v2=False,
            training: bool = True
    ) -> None:

        super().__init__()

        self.training = training
        self.resuetencoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.resuetencoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.resuetencoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.resuetencoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.pool2 = Pool['MAX', 3](kernel_size=2)
        self.pool3 = Pool['MAX', 3](kernel_size=2)
        self.pool4 = Pool['MAX', 3](kernel_size=2)

        self.gate2 = Gate(in_channels_up=feature_size, in_channels_down=2 * feature_size,
                          out_channels=feature_size)
        self.gate3 = Gate(in_channels_up=2 * feature_size, in_channels_down=4 * feature_size,
                          out_channels=2 * feature_size)
        self.gate4 = Gate(in_channels_up=4 * feature_size, in_channels_down=8 * feature_size,
                          out_channels=4 * feature_size)

        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swintransformer = SwinLayer(dim=8 * feature_size,
                                         num_heads=3, depth=depth,
                                         window_size=window_size, normalize=normalize,
                                         downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample,
                                                                                                           str) else downsample)
        self.CSA = CSA(in_chans=16 * feature_size, img_size=img_size)
        self.SSA = SSA(hidden_size=16 * feature_size, img_size=img_size, num_heads=3)

        self.dt4 = DepTran(in_channels=16 * feature_size, out_channels=16 * feature_size)
        self.dt3 = DepTran(in_channels=8 * feature_size, out_channels=8 * feature_size)
        self.dt2 = DepTran(in_channels=4 * feature_size, out_channels=4 * feature_size)
        self.dt1 = DepTran(in_channels=2 * feature_size, out_channels=2 * feature_size)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decodercat = Outlayer(in_channels=feature_size,
                                   out_channels=3,
                                   activation='Softmax')

        self.decodercl_r = Outlayer(in_channels=feature_size,
                                    out_channels=1,
                                    activation='Sigmoid')

        self.decodere_seg = Outlayer(in_channels=feature_size,
                                     out_channels=2,
                                     activation='Softmax')

        self.deepsupervision2 = Outlayer(in_channels=feature_size * 2,
                                         out_channels=3,
                                         activation='Softmax')

        self.deepsupervision3 = Outlayer(in_channels=feature_size * 4,
                                         out_channels=3,
                                         activation='Softmax')

    def forward(self, x):
        x1 = self.resuetencoder1(x)
        x2 = self.resuetencoder2(x1)
        x2 = self.pool2(x2)
        x1 = self.gate2(x1, x2)
        x3 = self.resuetencoder3(x2)
        x3 = self.pool3(x3)
        x2 = self.gate3(x2, x3)
        x4 = self.resuetencoder4(x3)
        x4 = self.pool4(x4)
        x3 = self.gate4(x3, x4)
        x5 = self.swintransformer(x4.contiguous())

        x5 = self.CSA(x5)
        x5 = self.SSA(x5)

        up5 = self.dt4(x5)
        up4 = self.decoder5(up5, x4)
        up4 = self.dt3(up4)
        up3 = self.decoder4(up4, x3)
        up3 = self.dt2(up3)
        up2 = self.decoder3(up3, x2)
        up2 = self.dt1(up2)
        up1 = self.decoder2(up2, x1)

        seg_v = self.decodercat(up1)
        seg_e = self.decodere_seg(up1)
        reg = self.decodercl_r(up1)

        deep2 = self.deepsupervision2(up2)
        deep3 = self.deepsupervision3(up3)

        if self.training:
            return seg_v, reg, seg_e, deep2, deep3
            # return x5
        else:
            return seg_v


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 96
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    model = VSNet(img_size=image_size, in_channels=1, out_channels=3, training=False)
    # out, out2, out3, out4, out5 = model(x)
    out = model(x)
    print("out size: {}".format(out.size()))
    # print("out2 size: {}".format(out2.size()))
    # print("out3 size: {}".format(out3.size()))
    # print("out4 size: {}".format(out4.size()))
    # print("out5 size: {}".format(out5.size()))
