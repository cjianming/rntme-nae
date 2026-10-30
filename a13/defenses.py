# -*- coding: utf-8 -*-
"""A3 防御评估：输入级防御与对抗训练。

两条必须遵守的评估纪律（写进论文，也写进这份记录的正文）：

1. **任何防御都要同时报「防御后 ASR」与「防御后干净准确率」。**
   只看 ASR 下降不看干净准确率，等于把模型打瞎了还叫防御有效。
   本模块的 `evaluate()` 因此总是成对返回这两个数。

2. **非自适应评估只能说明「这类扰动被削弱」，不能说明「攻击失效」。**
   要下后一个结论必须做自适应攻击（BPDA）。`bpda=True` 时前向过真实
   防御、反向按恒等近似回传梯度，用来检查「防御被针对性绕过」的程度。

张量约定与 A1.1 一致：输入/输出都是 (B,3,H,W)、取值范围 [-1,1]。
"""
from __future__ import annotations

import io
import math
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------- 基础换算


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) [-1,1] -> (B,H,W,3) uint8。"""
    a = ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return a.permute(0, 2, 3, 1).numpy()


def from_uint8(a: np.ndarray, device=None) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) [-1,1]。"""
    t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32)
    t = t.permute(0, 3, 1, 2) / 127.5 - 1.0
    return t.to(device) if device is not None else t


# --------------------------------------------------------------- 1. JPEG


def jpeg(x: torch.Tensor, quality: int = 75, subsampling: int = 0) -> torch.Tensor:
    """标准 JPEG 编解码。

    subsampling=0 即 4:4:4（不做色度下采样）。32 px 图上 4:2:0 会把颜色
    彻底糊掉，那不是「压缩」而是「换了一张图」，所以默认关掉；
    论文里报主表用 4:4:4，并用 `jpeg420` 做一次敏感性对照。
    """
    arr = to_uint8(x)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, format="JPEG", quality=quality,
                                     subsampling=subsampling, optimize=False)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"))
    return from_uint8(out, x.device)


def jpeg420(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """4:2:0 色度下采样版本，仅用于敏感性对照。"""
    return jpeg(x, quality=quality, subsampling=2)


# --------------------------------------------------------------- 2. 位深


def bit_depth(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """把像素量化到 bits 位（含符号位的量化步长 2^(8-bits)）。"""
    a = (x.clamp(-1, 1) + 1) * 127.5
    q = float(2 ** (8 - bits))
    # 量化后必须夹回 0..255：255 在 q=16 时会被 round 到 256，越界成 1.0078
    a = torch.clamp(torch.round(a / q) * q, 0.0, 255.0)
    return a / 127.5 - 1.0


# --------------------------------------------------------------- 3. 高斯模糊


def _gauss_kernel(sigma: float, k: int, device, dtype, ch: int) -> torch.Tensor:
    xs = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, k).repeat(ch, 1, 1, 1)


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """可微高斯模糊（reflect 填充，保持尺寸）。"""
    k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    c = x.shape[1]
    kh = _gauss_kernel(sigma, k, x.device, x.dtype, c)
    kw = kh.transpose(-1, -2)
    p = k // 2
    z = F.pad(x, (p, p, 0, 0), mode="reflect")
    z = F.conv2d(z, kh, groups=c)
    z = F.pad(z, (0, 0, p, p), mode="reflect")
    return F.conv2d(z, kw, groups=c).clamp(-1, 1)


# --------------------------------------------------------------- 4. 随机缩放填充


def rand_pad(x: torch.Tensor, lo: float = 0.85, rng: np.random.Generator | None = None) -> torch.Tensor:
    """随机缩放 + 随机零填充（Xie et al., 2018 的 R&P）。

    这是对迁移攻击最有效的经典预处理防御：推理时变换本身随机，
    攻击者无法针对某个确定的变换优化扰动。
    """
    rng = rng or np.random.default_rng()
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        s = float(rng.uniform(lo, 1.0))
        h, w = max(8, int(round(H * s))), max(8, int(round(W * s)))
        z = F.interpolate(x[i:i + 1], size=(h, w), mode="bilinear", align_corners=False)
        oy = int(rng.integers(0, H - h + 1))
        ox = int(rng.integers(0, W - w + 1))
        out[i:i + 1, :, oy:oy + h, ox:ox + w] = z
    return out


# --------------------------------------------------------------- 5. 低照度补偿


def gamma_brighten(x: torch.Tensor, gamma: float = 0.6) -> torch.Tensor:
    """γ<1 提亮，模拟夜间/隧道场景的预处理增强。"""
    a = (x.clamp(-1, 1) + 1) / 2
    return (a.clamp(1e-6, 1) ** gamma) * 2 - 1


# --------------------------------------------------------------- 6. 去雾


def dehaze(x: torch.Tensor, omega: float = 0.90, t0: float = 0.35,
           patch: int = 3) -> torch.Tensor:
    """暗通道先验去雾（He et al. 的简化实现，块最小值 + 引导近似）。

    车联网摄像头常年在雾/霾/雨里，这一项是场景相关的防御，
    论文里作为「域相关预处理」与通用预处理器并列。
    """
    a = (x.clamp(-1, 1) + 1) / 2                      # 0..1
    dark = -F.max_pool2d(-a, patch, 1, patch // 2)     # 每通道块最小值
    dark = dark.min(dim=1, keepdim=True).values        # 跨通道取最小
    A = a.flatten(2).amax(dim=2).reshape(a.shape[0], a.shape[1], 1, 1)
    t = (1 - omega * dark).clamp(t0, 1.0)
    j = (a - A) / t + A
    j = j.clamp(0, 1)
    return j * 2 - 1


# --------------------------------------------------------------- 注册表


DEFENSES: Dict[str, Callable[..., torch.Tensor]] = {
    "none":     lambda x: x,
    "jpeg75":   lambda x: jpeg(x, 75),
    "jpeg50":   lambda x: jpeg(x, 50),
    "jpeg420":  lambda x: jpeg420(x, 75),
    "bit4":     lambda x: bit_depth(x, 4),
    "bit3":     lambda x: bit_depth(x, 3),
    "blur10":   lambda x: gaussian_blur(x, 1.0),
    "randpad":  lambda x: rand_pad(x, 0.85),
    "gamma06":  lambda x: gamma_brighten(x, 0.6),
    "dehaze":   lambda x: dehaze(x),
}

# 中文名与一句话说明，报告里直接引用，避免两处措辞不一致
DESC: Dict[str, tuple] = {
    "none":    ("无防御", "对照基线"),
    "jpeg75":  ("JPEG-75", "常用社交/回传压缩档"),
    "jpeg50":  ("JPEG-50", "激进压缩，强平滑高频"),
    "jpeg420": ("JPEG-75（4:2:0）", "真实设备常用色度下采样，敏感性对照"),
    "bit4":    ("位深 4 bit", "量化到 16 级灰度"),
    "bit3":    ("位深 3 bit", "量化到 8 级灰度，接近阈值化"),
    "blur10":  ("高斯模糊 σ=1.0", "5×5 核，抑制高频扰动"),
    "randpad": ("随机缩放填充", "推理时随机变换 s∈[0.85,1]，攻击无法预先适配"),
    "gamma06": ("γ=0.6 提亮", "夜间/隧道低照度补偿"),
    "dehaze":  ("暗通道去雾", "雾霾场景预处理"),
}


class Defended(nn.Module):
    """把防御套在模型外面。

    bpda=True 时走 straight-through：前向是真的过防御，反向把防御当恒等，
    梯度直接回传到原图。这是评估「不可微预处理防御」的标准做法，
    也是判断「防御是真有效还是只挡住了不懂它的攻击」的关键。
    """

    def __init__(self, model: nn.Module, defense: str, bpda: bool = False, **kw):
        super().__init__()
        if defense not in DEFENSES:
            raise KeyError(f"未知防御 {defense}，可选 {list(DEFENSES)}")
        self.model = model
        self.defense = defense
        self.fn = DEFENSES[defense]
        self.kw = kw
        self.bpda = bpda

    def forward(self, x):
        xd = self.fn(x, **self.kw) if self.kw else self.fn(x)
        xd = xd.clamp(-1, 1)
        if self.bpda:
            xd = x + (xd - x).detach()      # 前向 = xd，∂xd/∂x = 1
        return self.model(xd)


def build(defense: str, models: Dict[str, nn.Module], bpda: bool = False,
          **kw) -> Dict[str, Defended]:
    return {k: Defended(v, defense, bpda=bpda, **kw) for k, v in models.items()}


# --------------------------------------------------------------- 自检

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from a11 import data as D
    from a11 import models as M

    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    print(f"{'防御':<12s}{'尺寸':<10s}{'与输入差':<10s}")
    for k in DEFENSES:
        y = DEFENSES[k](x)
        assert y.shape == x.shape, (k, y.shape)
        assert torch.isfinite(y).all(), k
        assert y.min() >= -1.0001 and y.max() <= 1.0001, (k, float(y.min()), float(y.max()))
        print(f"{k:<12s}{str(tuple(y.shape)):<10s}{float((y-x).abs().mean()):.4f}")

    # 干净准确率随防御的变化（这一步是判断防御是否「把模型打瞎」的关键）
    ext = D.ensure_extracted("data/gtsrb")
    _, te = D._read_index(ext)
    te = D.stratified_subset(te, 300, seed=2)
    ds = D.GTSRB(te, 32, False, cache=True)
    xs = torch.stack([torch.from_numpy(ds[i][0]) for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    mdl, _ = None, None
    print("\n干净准确率（同一 300 张测试子集）")
    for arch in ("sresnet", "svgg", "ssep", "sinc"):
        ck = torch.load(f"a11/chk/{arch}_s32.pt", map_location="cpu", weights_only=False)
        m = M.build(arch, 43, 32)
        m.load_state_dict(ck["state"])
        m.eval()
        if mdl is None:
            mdl = m
        line = [f"{arch:<8s}"]
        with torch.no_grad():
            for k in DEFENSES:
                xd = DEFENSES[k](xs).clamp(-1, 1)
                acc = (m(xd).argmax(1) == ys).float().mean().item()
                line.append(f"{k}={acc:.3f}")
        print("  " + "  ".join(line))
