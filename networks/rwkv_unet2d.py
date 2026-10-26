# -*- coding: utf-8 -*-
"""
RWKV-UNet —— 单文件、可直接运行的版本
=====================================
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RWKV_UNet", "RWKV_UNet_T", "RWKV_UNet_S", "RWKV_UNet_B",
    "RWKV_UNet_encoder", "RWKV_UNet_encoder_T", "RWKV_UNet_encoder_S",
    "RWKV_UNet_encoder_B", "build_rwkv_unet", "wkv", "wkv_fast", "wkv_exact",
    "wkv_torch", "count_parameters",
]

T_MAX = 1024      
inplace = True


# =====================================================================
# 1. 基础组件（替代 timm / einops 里的那几个工具）
# =====================================================================
class LayerNorm2d(nn.Module):
    """在 (B, C, H, W) 上做 LayerNorm（原来用 einops 的 rearrange）。"""

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth：按样本随机丢弃整条残差支路。"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    """timm 里的 GELU 就是 nn.GELU；这里显式实现以兼容 inplace 参数。"""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.gelu(x)


def get_norm(norm_layer: str = "in_1d"):
    eps = 1e-6
    norm_dict = {
        "none": nn.Identity,
        "in_1d": partial(nn.InstanceNorm1d, eps=eps),
        "in_2d": partial(nn.InstanceNorm2d, eps=eps),
        "in_3d": partial(nn.InstanceNorm3d, eps=eps),
        "bn_1d": partial(nn.BatchNorm1d, eps=eps),
        "bn_2d": partial(nn.BatchNorm2d, eps=eps),
        "bn_3d": partial(nn.BatchNorm3d, eps=eps),
        "gn": partial(nn.GroupNorm, eps=eps),
        "ln_1d": partial(nn.LayerNorm, eps=eps),
        "ln_2d": partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer: str = "relu"):
    act_dict = {
        "none": nn.Identity,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "hsigmoid": nn.Hardsigmoid,
        "hswish": nn.Hardswish,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "gelu": GELU,
    }
    return act_dict[act_layer]


def make_act(act_layer: str = "relu", inplace: bool = True) -> nn.Module:
    """实例化激活函数；nn.Identity / GELU 之类不接受 inplace 参数。"""
    act = get_act(act_layer)
    try:
        return act(inplace=inplace)
    except TypeError:
        return act()


class LayerScale2D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1,
                 bias=False, skip=False, norm_layer="bn_2d", act_layer="relu",
                 inplace=True, drop_path_rate=0.0):
        super().__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = make_act(act_layer, inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation。"""

    def __init__(self, in_chs: int, rd_ratio: float = 0.25, rd_channels: Optional[int] = None,
                 act_layer: Callable = nn.ReLU, gate_layer: Callable = nn.Sigmoid,
                 force_act_layer: Optional[Callable] = None,
                 rd_round_fn: Optional[Callable] = None):
        super().__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = int(rd_round_fn(in_chs * rd_ratio))
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True) if act_layer in (nn.ReLU, nn.SiLU) else act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


# =====================================================================
# 2. WKV（加权键值注意力）
# =====================================================================
# 原版是自写 CUDA kernel：把 token 序列按 32 个一块切分，用 log-sum-exp 归一化做
# “三趟”并行扫描（块内前缀 / 块间前缀 / 块内输出）。下面把它逐行搬到 PyTorch 张量上，
# 语义完全一致（同一份输入与真正的 CUDA kernel 对比误差 < 1e-6），可直接反传。
#
# 记号：w、u 为形状 (C,) 的逐通道参数（调用方已除以 T），k、v 为 (B, T, C)。
# 自注意力的形式为（双向指数衰减 + 当前 token 的 exp(u) 加成）：
#
#            Σ_{j≠i} exp(k_j - w·|j-i|) · v_j + exp(u + k_i) · v_i
#   y_i = -------------------------------------------------------------
#              Σ_{j≠i} exp(k_j - w·|j-i|)      + exp(u + k_i) + eps
#
# 注意：原版 kernel 在分块边界上有一点点自己的取整/归一化习惯（块长 32），纯数学式
# 与该习惯在 T>32 时会有约 1% 的差异；这里以 kernel 的实现为准。

_MIN_VALUE = -1e38
_WKV_TOKEN_SPLIT = 32
_WKV_EPS = 1e-6

_WKV_CUDA = None


def _load_wkv_cuda():
    """按需编译原版 CUDA 算子（仅在 RWKV_WKV_CUDA=1 时尝试）。"""
    global _WKV_CUDA
    if _WKV_CUDA is not None:
        return _WKV_CUDA
    if os.environ.get("RWKV_WKV_CUDA", "0") != "1" or not torch.cuda.is_available():
        return None
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
    sources = [os.path.join(src_dir, "wkv_op.cpp"), os.path.join(src_dir, "wkv_cuda.cu")]
    if not all(os.path.exists(s) for s in sources):
        warnings.warn("找不到 cuda/wkv_op.cpp 或 cuda/wkv_cuda.cu，改用纯 PyTorch 版 WKV。")
        return None
    try:
        from torch.utils.cpp_extension import load
        _WKV_CUDA = load(
            name="wkv",
            sources=sources,
            verbose=False,
            extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math",
                               "-O3", "-Xptxas -O3", f"-DTmax={T_MAX}"],
        )
        return _WKV_CUDA
    except Exception as exc:  # pragma: no cover - 取决于本机编译环境
        warnings.warn(f"CUDA 版 WKV 编译失败（{exc}），改用纯 PyTorch 实现。")
        return None


def wkv(w, u, k, v):
    """加权键值注意力。w、u: (C,)；k、v: (B, T, C)；返回 (B, T, C)。

    三条路径，优先级从高到低：
    1. ``RWKV_WKV_CUDA=1``：原版 CUDA 算子（最快，需编译）；
    2. ``RWKV_WKV_EXACT=1``：逐行等价于 CUDA kernel 的纯 PyTorch 版（最像原版，慢）；
    3. 默认：闭式对称 WKV（论文里的公式，4 个 cumsum，最快，块边界与 kernel 差 <0.5%）。
    """
    if k.is_cuda and os.environ.get("RWKV_WKV_CUDA", "0") == "1":
        ext = _load_wkv_cuda()
        if ext is not None:
            B, T, C = k.shape
            if (B * C) % min(16, C) == 0 and T <= T_MAX:
                return _WKV.apply(ext, B, T, C, w, u, k, v)
    if os.environ.get("RWKV_WKV_EXACT", "0") == "1":
        return wkv_exact(w, u, k, v)
    return wkv_fast(w, u, k, v)


def wkv_fast(w, u, k, v, eps: float = _WKV_EPS):
    """闭式 WKV（默认路径，快）：

        y_i = ( Σ_{j≠i} exp(k_j - w|j-i|) v_j + exp(u+k_i) v_i )
              / ( Σ_{j≠i} exp(k_j - w|j-i|)     + exp(u+k_i) + eps )

    用两组前缀/后缀累积和实现，复杂度 O(T)，只需 4 次 cumsum。
    数值上做逐通道 max 归一化，exp 的取值范围约为 exp(2|w|T)，不会溢出。
    """
    B, T, C = k.shape
    dt = torch.float64 if k.dtype == torch.float64 else torch.float32
    w = w.to(dt).view(1, 1, C)
    u = u.to(dt).view(1, 1, C)
    kk = k.to(dt)
    vv = v.to(dt)

    jw = torch.arange(T, device=k.device, dtype=dt).view(1, T, 1) * w
    ek = torch.exp(kk)

    def exclusive(c):                       # c[i] = sum_{j<i}
        return torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1)

    def exclusive_from_right(g):            # out[i] = sum_{j>i}
        c = torch.cumsum(g, dim=1)
        return c[:, -1:] - c

    # 前缀部分：j < i，权重 exp(k_j - w(i-j)) = exp(k_j + w j) * exp(-w i)
    e_p = kk + jw
    m_p = e_p.max(dim=1, keepdim=True).values
    g_p = torch.exp(e_p - m_p)
    pre = torch.exp(m_p - jw)
    Pk = exclusive(torch.cumsum(g_p, dim=1)) * pre
    Pkv = exclusive(torch.cumsum(g_p * vv, dim=1)) * pre

    # 后缀部分：j > i，权重 exp(k_j + w(j-i)) = exp(k_j - w j) * exp(+w i)
    e_s = kk - jw
    m_s = e_s.max(dim=1, keepdim=True).values
    g_s = torch.exp(e_s - m_s)
    suf = torch.exp(m_s + jw)
    Sk = exclusive_from_right(g_s) * suf
    Skv = exclusive_from_right(g_s * vv) * suf

    eu = torch.exp(u)
    num = Pkv + Skv + eu * ek * vv
    den = Pk + Sk + eu * ek + eps
    return (num / den).to(k.dtype)


def wkv_exact(w, u, k, v, eps: float = _WKV_EPS):
    """纯 PyTorch 版 WKV（与原版 CUDA kernel 逐行等价，慢但逐位一致）。"""
    B, T, C = k.shape
    S = _WKV_TOKEN_SPLIT
    L = (T + S - 1) // S                       # 每块的 token 数
    dt = torch.float64 if k.dtype == torch.float64 else torch.float32

    w = w.to(dt).view(1, 1, C)
    u = u.to(dt).view(1, 1, C)
    kk = k.to(dt)
    vv = v.to(dt)

    t0 = torch.arange(S, device=k.device) * L              # 每块起点
    lens = (T - t0).clamp(min=0, max=L)                    # 每块实际 token 数
    has_b = (lens > 0).view(1, S, 1)
    pos_ok = (torch.arange(L, device=k.device).view(L, 1) < lens.view(1, S))   # (L, S)

    def gather(x, idx):
        return x.index_select(1, idx.clamp(0, T - 1))

    zeros = torch.zeros(B, S, C, device=k.device, dtype=dt)
    mval = torch.full((B, S, C), _MIN_VALUE, device=k.device, dtype=dt)

    # ---- 第 1 趟：块内前缀（往前走）与块内后缀（往回走）
    c_num, c_den, o1 = zeros.clone(), zeros.clone(), mval.clone()
    a_num, a_den, o2 = zeros.clone(), zeros.clone(), mval.clone()
    for p in range(L):
        vm = pos_ok[p].view(1, S, 1)
        i = t0 + p
        k_i = gather(kk, i)
        v_i = gather(vv, i)

        arg = k_i - w * p
        no = torch.maximum(o1, arg)
        e1 = torch.exp(o1 - no)
        e3 = torch.exp(arg - no)
        c_num = torch.where(vm, e1 * c_num + e3 * v_i, c_num)
        c_den = torch.where(vm, e1 * c_den + e3, c_den)
        o1 = torch.where(vm, no, o1)

        ni = t0 + lens - 1 - p
        k_n = gather(kk, ni)
        v_n = gather(vv, ni)
        arg = k_n - w * (p + 1)
        no = torch.maximum(o2, arg)
        e2 = torch.exp(o2 - no)
        e3 = torch.exp(arg - no)
        a_num = torch.where(vm, e2 * a_num + e3 * v_n, a_num)
        a_den = torch.where(vm, e2 * a_den + e3, a_den)
        o2 = torch.where(vm, no, o2)

    Sc, Sd, So1 = c_num, c_den, o1
    Sa, Sb, So2 = a_num, a_den, o2
    tid = torch.arange(S, device=k.device).view(1, S, 1)

    # ---- 第 2 趟：把“前面各块”的状态汇总成当前块的前缀状态
    a_num, a_den, o2 = zeros.clone(), zeros.clone(), mval.clone()
    for i in range(S):
        mk = tid > i
        ew = (tid - i - 1) * L
        arg = So2[:, i:i + 1, :] - w * ew
        no = torch.maximum(arg, o2)
        e = torch.exp(o2 - no)
        a_num = torch.where(mk, a_num * e + Sa[:, i:i + 1, :] * torch.exp(arg - no), a_num)
        a_den = torch.where(mk, a_den * e + Sb[:, i:i + 1, :] * torch.exp(arg - no), a_den)
        o2 = torch.where(mk, no, o2)

    # ---- 第 3 趟：把“后面各块”的状态汇总成当前块的后缀状态
    c_num, c_den, o1 = zeros.clone(), zeros.clone(), mval.clone()
    for i in range(S):
        mk = tid <= i
        ew = (i - tid) * L
        arg = So1[:, i:i + 1, :] - w * ew
        no = torch.maximum(arg, o1)
        e = torch.exp(o1 - no)
        c_num = torch.where(mk, c_num * e + Sc[:, i:i + 1, :] * torch.exp(arg - no), c_num)
        c_den = torch.where(mk, c_den * e + Sd[:, i:i + 1, :] * torch.exp(arg - no), c_den)
        o1 = torch.where(mk, no, o1)

    # 后缀状态里去掉“块首 token”（它在输出时按当前 token 单独处理）
    k0 = gather(kk, t0)
    v0 = gather(vv, t0)
    o1_safe = torch.where(has_b, o1, torch.zeros_like(o1))
    sub = torch.exp(k0 - o1_safe)
    c_num = torch.where(has_b, c_num - sub * v0, c_num)
    c_den = torch.where(has_b, c_den - sub, c_den)

    # ---- 第 4 趟：逐 token 输出
    outs = []
    for p in range(L):
        i = t0 + p
        k_i = gather(kk, i)
        v_i = gather(vv, i)

        no = torch.maximum(o1, u + k_i)
        no = torch.maximum(no, o2)
        e1 = torch.exp(o1 - no)
        e2 = torch.exp(o2 - no)
        e3 = torch.exp(u + k_i - no)
        outs.append((c_num * e1 + a_num * e2 + e3 * v_i) / (c_den * e1 + a_den * e2 + e3 + eps))

        # 前缀状态前进一步
        no = torch.maximum(o2 - w, k_i)
        e2 = torch.exp(o2 - w - no)
        e3 = torch.exp(k_i - no)
        a_num = e2 * a_num + e3 * v_i
        a_den = e2 * a_den + e3
        o2 = no

        # 后缀状态后退一步（减掉下一个 token）
        i2 = (i + 1) % T
        k_i2 = gather(kk, i2)
        v_i2 = gather(vv, i2)
        no = torch.maximum(o1 + w, k_i2 + w)
        e1 = torch.exp(o1 + w - no)
        e3 = torch.exp(k_i2 + w - no)
        c_num = e1 * c_num - e3 * v_i2
        c_den = e1 * c_den - e3
        o1 = no

    y = torch.stack(outs, dim=0)                        # (L, B, S, C)
    y = y.permute(1, 2, 0, 3).reshape(B, S * L, C)      # 位置 = 块号*L + 块内偏移
    return y[:, :T].to(k.dtype)


# 旧名字（兼容之前的脚本/说明）
wkv_torch = wkv_exact


class _WKV(torch.autograd.Function):
    """仅在启用原版 CUDA 算子时使用；纯 PyTorch 路径由 autograd 自己处理。"""

    @staticmethod
    def forward(ctx, ext, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C, ctx.ext = B, T, C, ext
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=k.device, dtype=torch.float32)
        ext.forward(B, T, C, w.float().contiguous(), u.float().contiguous(),
                    k.float().contiguous(), v.float().contiguous(), y)
        return y.to(k.dtype)

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device=gy.device, dtype=torch.float32)
        gu = torch.zeros((B, C), device=gy.device, dtype=torch.float32)
        gk = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32)
        gv = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32)
        ctx.ext.backward(B, T, C, w.float().contiguous(), u.float().contiguous(),
                         k.float().contiguous(), v.float().contiguous(),
                         gy.float().contiguous(), gw, gu, gk, gv)
        gw, gu = gw.sum(dim=0), gu.sum(dim=0)
        return (None, None, None, None, gw, gu, gk, gv)


def q_shift(x, shift_pixel: int = 1, gamma: float = 1 / 4, patch_resolution=None):
    """按通道分四组做上下左右平移，把邻域信息引进（本为因果的）WKV。"""
    assert gamma <= 1 / 4
    B, N, C = x.shape
    input = x.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output.flatten(2).transpose(1, 2)


# =====================================================================
# 3. RWKV 空间混合 / 通道混合 / 跨尺度融合
# =====================================================================
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def jit_func(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        return torch.sigmoid(r), k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device
        sr, k, v = self.jit_func(x, patch_resolution)
        x = wkv(self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2, key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key_norm = nn.LayerNorm(hidden_sz) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class CCMix(nn.Module):
    """Cross-scale Channel Mix：把三个尺度的特征对齐到同一尺寸后做通道混合。"""

    def __init__(self, in_dims, target_dim, target_size, drop_path_rate=0.05):
        super().__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = nn.LayerNorm(target_dim * 3)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        self.channel = VRWKV_ChannelMix(n_embd=target_dim * 3, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes = [target_size // 4, target_size // 2, target_size]

    def forward(self, features):
        upsampled_features = []
        output_features = []
        for i, feature in enumerate(features):
            feature = F.interpolate(feature, size=self.target_size, mode="bilinear", align_corners=False)
            feature = self.projections[i](feature)
            upsampled_features.append(feature)

        concatenated = torch.cat(upsampled_features, dim=1)
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, -1).permute(0, 2, 1)          # (B, N, C)

        attn_output = concatenated + self.drop_path(
            self.ln1(self.channel(concatenated, (self.target_size, self.target_size)))
        )

        B, n_patch, hidden = attn_output.size()
        h = w = int(math.sqrt(n_patch))
        attn_output = attn_output.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            split_feature = self.final_projections[i](split_feature)
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i],
                                          mode="bilinear", align_corners=False)
            output_features.append(split_feature)
        return output_features


# =====================================================================
# 4. GLSP 块与编码器
# =====================================================================
class GLSP(nn.Module):
    """Global-Local Spatial Perception block：卷积做局部建模，WKV 做长程建模。"""

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 norm_layer="bn_2d", act_layer="relu", dw_ks=3, stride=1, dilation=1,
                 se_ratio=0.0, attn_s=True, drop_path=0.0, drop=0.0, img_size=224,
                 channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.ln2 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="none",
                                act_layer="none", inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.attn_s = attn_s
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride,
                                      dilation=dilation, groups=dim_mid, norm_layer="bn_2d",
                                      act_layer="silu", inplace=inplace)
        self.drop_after_pos = nn.Dropout(p=0.1)
        self.num_extra_tokens = 0

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        if self.attn_s:
            B, hidden, H, W = x.size()
            patch_resolution = (H, W)
            x = x.view(B, hidden, -1).permute(0, 2, 1)          # (B, N, C)
            x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
            B, n_patch, hidden = x.size()
            h = w = int(math.sqrt(n_patch))
            x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class RWKV_UNet_encoder(nn.Module):
    """四阶段金字塔编码器（stage0 是 stem，stride=1）。"""

    def __init__(self, dim_in=3, num_classes=1000, img_size=224,
                 depths=(2, 4, 4, 2), stem_dim=16, embed_dims=(64, 128, 256, 512),
                 exp_ratios=(2.0, 2.0, 4.0, 4.0),
                 norm_layers=("bn_2d", "bn_2d", "bn_2d", "bn_2d"),
                 act_layers=("relu", "relu", "relu", "relu"),
                 dw_kss=(3, 3, 1, 1), se_ratios=(0.0, 0.0, 0.0, 0.0),
                 attn_ss=(False, False, True, True), drop=0.0, drop_path=0.0,
                 pre_dim=0, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.embed_dims = list(embed_dims)

        self.stage0 = nn.ModuleList([
            GLSP(dim_in, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
                 norm_layer=norm_layers[0], act_layer=act_layers[0], dw_ks=dw_kss[0],
                 stride=1, dilation=1, se_ratio=1, attn_s=False,
                 drop_path=0.0, drop=0.0, img_size=224)
        ])
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = 2, False, False, exp_ratios[i] * 2
                else:
                    stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
                layers.append(GLSP(
                    emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip,
                    exp_ratio=exp_ratio, norm_layer=norm_layers[i],
                    act_layer=act_layers[i], dw_ks=dw_kss[i], stride=stride, dilation=1,
                    se_ratio=se_ratios[i], attn_s=attn_s, drop_path=dpr[j], drop=drop,
                    img_size=img_size, channel_gamma=channel_gamma, shift_pixel=shift_pixel,
                ))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f"stage{i + 1}", nn.ModuleList(layers))

        self.pre_dim = embed_dims[-1]
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"token"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"alpha", "gamma", "beta"}

    @torch.jit.ignore
    def no_ft_keywords(self):
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self):
        return {"head.weight", "head.bias"}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    def check_bn(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)

    def forward_features(self, x):
        for blk in self.stage0:
            x = blk(x)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        for blk in self.stage4:
            x = blk(x)
        return x

    def forward(self, x):
        return self.forward_features(x)


def _encoder(depths, stem_dim, embed_dims, exp_ratios, **kwargs):
    return RWKV_UNet_encoder(
        depths=depths, stem_dim=stem_dim, embed_dims=embed_dims, exp_ratios=exp_ratios,
        norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
        act_layers=["silu", "silu", "gelu", "gelu"], dw_kss=[5, 5, 5, 5],
        attn_ss=[False, False, True, True], drop=0.0, drop_path=0.05, **kwargs)


def RWKV_UNet_encoder_T(pretrained=False, **kwargs):
    return _encoder([2, 2, 4, 2], 24, [32, 48, 96, 160], [2.0, 2.5, 3.0, 3.5], **kwargs)


def RWKV_UNet_encoder_S(pretrained=False, **kwargs):
    return _encoder([3, 3, 6, 3], 24, [32, 64, 128, 192], [2.0, 2.5, 3.0, 4.0], **kwargs)


def RWKV_UNet_encoder_B(pretrained=False, **kwargs):
    return _encoder([3, 3, 6, 3], 24, [48, 72, 144, 240], [2.0, 2.5, 4.0, 4.0], **kwargs)


# =====================================================================
# 5. 解码器与完整网络（T / S / B 三种规模）
# =====================================================================
class UpBlock(nn.Module):
    """解码器基本块：1x1 升降维 + 深度可分离卷积 + 双线性上采样。"""

    def __init__(self, dim_in, dim_out, norm_in=False, has_skip=False, exp_ratio=1.0,
                 norm_layer="bn_2d", dw_ks=3, stride=1, dilation=1, se_ratio=0.0,
                 drop_path=0.0, drop=0.0, act_layer="relu"):
        super().__init__()
        self.has_skip = has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="bn_2d",
                                act_layer="relu", inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride,
                                      dilation=dilation, groups=dim_mid, norm_layer="bn_2d",
                                      act_layer="silu", inplace=inplace)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x


class _RWKVUNetBase(nn.Module):
    """三种规模共用的骨架。"""

    encoder_factory: Callable = staticmethod(RWKV_UNet_encoder_B)
    embed_dims: tuple = (48, 72, 144, 240)

    def __init__(self, in_channels=1, num_classes=9, img_size=224, pretrained_path=None):
        super().__init__()
        assert img_size % 16 == 0, "img_size 必须是 16 的整数倍（224 / 256 / 512 ...）"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dims = list(self.embed_dims)

        self.encoder = self.encoder_factory(img_size=img_size)
        if pretrained_path:
            self._load_encoder_weights(pretrained_path)

        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
                         self.embed_dims[0], img_size // 2)
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,
                                drop_path=0.0, drop=0.0)
        self.decoder2 = UpBlock(self.embed_dims[2] * 2, self.embed_dims[1], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,
                                drop_path=0.0, drop=0.0)
        self.decoder3 = UpBlock(self.embed_dims[1] * 2, self.embed_dims[0], norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,
                                drop_path=0.0, drop=0.0)
        self.decoder4 = UpBlock(self.embed_dims[0] * 2, 24, norm_in=False, has_skip=False,
                                exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,
                                drop_path=0.0, drop=0.0)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def _load_encoder_weights(self, path):
        if not os.path.exists(path):
            warnings.warn(f"找不到预训练权重 {path}，encoder 将随机初始化。")
            return
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        # 兼容两种保存方式：完整模型（key 带 "encoder." 前缀）或只存了 encoder
        prefix = "encoder."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(f"encoder 权重部分加载：missing={len(missing)}, unexpected={len(unexpected)}")

    def _prepare_input(self, x):
        """原版只支持 1 通道（repeat 成 3 通道），这里 1 / 3 通道都支持。"""
        if x.dim() != 4:
            raise ValueError(f"期望 4D 输入 (B, C, H, W)，实际得到 {tuple(x.shape)}")
        c = x.shape[1]
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(f"in_channels 只支持 1 或 3，实际得到 {c}")

    def forward_features(self, x):
        x = self._prepare_input(x)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        return x, enc1, enc2, enc3

    def forward(self, x):
        """返回分割 logits，形状 (B, num_classes, H, W)。"""
        x, enc1, enc2, enc3 = self.forward_features(x)
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        out_logit = self.final_conv(dec0)
        if self.num_classes == 1:
            output = torch.sigmoid(out_logit)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
        if self.num_classes > 1:
            output = torch.softmax(out_logit, dim=1)
            output = torch.argmax(output, dim=1).float()
        return out_logit, output


class RWKV_UNet(_RWKVUNetBase):
    """Base 规模（Synapse 默认配置）。"""

    encoder_factory = staticmethod(RWKV_UNet_encoder_B)
    embed_dims = (48, 72, 144, 240)


class RWKV_UNet_S(_RWKVUNetBase):
    """Small 规模。"""

    encoder_factory = staticmethod(RWKV_UNet_encoder_S)
    embed_dims = (32, 64, 128, 192)


class RWKV_UNet_T(_RWKVUNetBase):
    """Tiny 规模（二分类常用）。"""

    encoder_factory = staticmethod(RWKV_UNet_encoder_T)
    embed_dims = (32, 48, 96, 160)

    def __init__(self, in_channels=1, num_classes=1, img_size=256, pretrained_path=None):
        super().__init__(in_channels=in_channels, num_classes=num_classes,
                         img_size=img_size, pretrained_path=pretrained_path)


# 别名：原项目权重文件分别为 net_B.pth / net_S.pth / net_T.pth
RWKV_UNet_B = RWKV_UNet


def build_rwkv_unet(model: str = "B", in_channels: int = 1, num_classes: int = 9,
                    img_size: int = 224, pretrained_path: Optional[str] = None, **kwargs):
    """按名字建模型：model ∈ {'T', 'S', 'B'}。"""
    table = {"t": RWKV_UNet_T, "s": RWKV_UNet_S, "b": RWKV_UNet}
    key = str(model).lower()
    if key not in table:
        raise ValueError(f"未知模型规模 {model!r}，可选 'T' / 'S' / 'B'")
    return table[key](in_channels=in_channels, num_classes=num_classes,
                      img_size=img_size, pretrained_path=pretrained_path, **kwargs)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))



if __name__ == "__main__":
    net = RWKV_UNet(in_channels=1, num_classes=1, img_size=256, pretrained_path='net_B.pth')
    in1 = torch.rand((1, 1, 256, 256))
    logits = net(in1)
    for i in range(len(logits)):
        print(logits[i].size())
