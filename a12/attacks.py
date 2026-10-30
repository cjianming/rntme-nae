# -*- coding: utf-8 -*-
"""目标检测任务的对抗攻击：I-FGSM / NAE / DI-NI-FGSM / RNTME-NAE（检测版）。

与分类版的差别有三处，必须写进论文：

1. **损失不同**。分类是单模型交叉熵；检测是「分类损失 + 框回归损失」之和，
   由各检测器自己的加权系数决定（Faster R-CNN 用 1:1，SSD 用 hard negative
   mining 后的分类损失 + smooth-L1）。
2. **几何变换必须同步变换标注框**。分类任务把猫旋转一下还是猫，检测任务旋转
   图像却不旋转框，loss 就完全错了。所以这里每个样本的缩放/平移参数都会
   同时作用到图像（grid_sample）与 GT 框（warp_boxes）。
3. **损失尺度跨模型不可比**。检测器的总损失从 2 到 10 不等，直接取倒数加权会
   让损失大的模型独占权重。因此动态权重用「相对损失」L_i / L_i^(0)（首步值）
   归一化后再取倒数，其余（温度、EMA、L1 归一化、高斯平滑）与论文式(7)-(10) 一致。
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .detectors import Detector


# ------------------------------------------------------------ 基础算子
def l1_norm(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    d = g.flatten(1).abs().sum(1).clamp_min(eps)
    return g / d.view(-1, *([1] * (g.dim() - 1)))


def gauss_kernel(sigma: float, k: int | None = None) -> torch.Tensor:
    if k is None:
        k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    xs = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def smooth(g: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return g
    k = gauss_kernel(sigma)
    ks, c = k.numel(), g.shape[1]
    kh = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    kw = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
    p = ks // 2
    g = F.conv2d(F.pad(g, (p, p, 0, 0), mode="reflect"), kh, groups=c)
    g = F.conv2d(F.pad(g, (0, 0, p, p), mode="reflect"), kw, groups=c)
    return g


# ------------------------------------------------------------ 同步变换
def rand_geo(x: torch.Tensor, s_lo: float, s_hi: float, shift: float,
             gen: torch.Generator | None = None):
    """随机缩放 + 平移。返回 (变换后的图, s, tx, ty)，s/tx/ty 均为 (n,) 逐样本。

    theta = [[1/s, 0, tx], [0, 1/s, ty]] 是「输出坐标 → 输入坐标」的映射，
    故 s<1 表示内容被缩小、视野变宽。tx/ty 为归一化坐标单位。
    """
    n = x.shape[0]
    dev, dt = x.device, x.dtype
    s = torch.empty(n, device=dev, dtype=dt).uniform_(s_lo, s_hi, generator=gen)
    amp = shift * (1.0 - s) if s_lo < s_hi else 0.0
    tx = torch.empty(n, device=dev, dtype=dt).uniform_(-1, 1, generator=gen) * amp
    ty = torch.empty(n, device=dev, dtype=dt).uniform_(-1, 1, generator=gen) * amp
    inv = (1.0 / s).unsqueeze(1)
    zero = torch.zeros(n, 1, device=dev, dtype=dt)
    theta = torch.cat([inv, zero, tx.unsqueeze(1), zero, inv, ty.unsqueeze(1)], 1).view(n, 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return out, s, tx, ty


def warp_boxes(boxes: torch.Tensor, s: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor,
               W: int, H: int) -> torch.Tensor:
    """把输入图像坐标系里的框，搬到 rand_geo 输出图像的坐标系里。

    由 o_norm = s * (i_norm - t) 反解，像素坐标下为
        o_x = s*x + (1-s)*W/2 - s*tx*W/2
        o_y = s*y + (1-s)*H/2 - s*ty*H/2
    宽高按 s 缩放。左右/下界越界的框裁到图内；完全出界的框会被压成退化框，
    后续由调用方按面积过滤。

    注意：本函数按「单张图 + 该图全部框」处理，s/tx/ty 允许是 0 维标量
    （逐图的调用方式）或长度 1 的一维张量。
    """
    s = s.reshape(1)
    tx = tx.reshape(1)
    ty = ty.reshape(1)
    b = boxes.clone()
    b[:, [0, 2]] = b[:, [0, 2]] * s.unsqueeze(1) + (1 - s).unsqueeze(1) * W / 2 \
        - s.unsqueeze(1) * tx.unsqueeze(1) * W / 2
    b[:, [1, 3]] = b[:, [1, 3]] * s.unsqueeze(1) + (1 - s).unsqueeze(1) * H / 2 \
        - s.unsqueeze(1) * ty.unsqueeze(1) * H / 2
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
    return b


def rand_color(x: torch.Tensor, bright: float, contrast: float, blur_sigma: float,
               gen: torch.Generator | None = None) -> torch.Tensor:
    n = x.shape[0]
    dev, dt = x.device, x.dtype
    out = x
    if bright > 0:
        f = torch.empty(n, 1, 1, 1, device=dev, dtype=dt).uniform_(1 - bright, 1 + bright, generator=gen)
        out = out * f
    if contrast > 0:
        mu = out.mean(dim=(1, 2, 3), keepdim=True)
        f = torch.empty(n, 1, 1, 1, device=dev, dtype=dt).uniform_(1 - contrast, 1 + contrast, generator=gen)
        out = (out - mu) * f + mu
    if blur_sigma > 0:
        k = gauss_kernel(blur_sigma, 5).to(dev, dt)
        ks, c = k.numel(), out.shape[1]
        kh = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
        kw = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
        p = ks // 2
        out = F.conv2d(F.pad(out, (p, p, 0, 0), mode="reflect"), kh, groups=c)
        out = F.conv2d(F.pad(out, (0, 0, p, p), mode="reflect"), kw, groups=c)
    return out.clamp(0, 1)


def _finish(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x_adv, x - eps, x + eps).clamp(0, 1)


# ------------------------------------------------------------ 集成梯度
def _one_grad(d: Detector, x: torch.Tensor, targets, seed: int | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (loss 标量, 对 x 的梯度)。loss 用于动态权重。"""
    x = x.detach().clone().requires_grad_(True)
    L = d.loss_sum(x, targets, seed)
    g, = torch.autograd.grad(L, x)
    return L.detach(), g.detach()


def _filter_targets(targets, boxes_new, min_area: float = 16.0):
    """变换后把退化框剔掉，避免除零与无效回归目标。"""
    out = []
    for t, b in zip(targets, boxes_new):
        wh = (b[:, 2:] - b[:, :2]).clamp_min(0)
        keep = (wh[:, 0] * wh[:, 1]) >= min_area
        out.append({"boxes": b[keep], "labels": t["labels"][keep]})
    return out


# ------------------------------------------------------------ 各攻击
def ifgsm(dets: Dict[str, Detector], x, targets, cfg) -> torch.Tensor:
    eps, alpha, T = cfg["eps"], cfg["alpha"], cfg["T"]
    name = cfg.get("source") or next(iter(dets))
    x_adv = x.clone()
    for t in range(T):
        _, g = _one_grad(dets[name], x_adv, targets, cfg.get("seed", 0) + t)
        x_adv = _finish(x_adv + alpha * g.sign(), x, eps)
    return x_adv.detach()


def nae(dets: Dict[str, Detector], x, targets, cfg) -> torch.Tensor:
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source") or next(iter(dets))
    x_adv = x.clone()
    v = torch.zeros_like(x)
    for t in range(T):
        x_nes = x_adv + mu * v
        _, g = _one_grad(dets[name], x_nes, targets, cfg.get("seed", 0) + t)
        v = mu * v + l1_norm(g)
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def di_ni_fgsm(dets: Dict[str, Detector], x, targets, cfg) -> torch.Tensor:
    """DI（随机缩放 + 平移，框同步）+ NI，单模型。检测任务里 DI 必须带框变换。"""
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source") or next(iter(dets))
    H, W = x.shape[-2], x.shape[-1]
    x_adv = x.clone()
    v = torch.zeros_like(x)
    gen = torch.Generator(device=x.device).manual_seed(cfg.get("seed", 0))
    for t in range(T):
        x_nes = x_adv + mu * v
        xt, s, tx, ty = rand_geo(x_nes, 0.85, 1.0, 0.06, gen)
        if cfg.get("warp_boxes", True):
            nb = [warp_boxes(tt["boxes"], s[i], tx[i], ty[i], W, H)
                  for i, tt in enumerate(targets)]
            tg = _filter_targets(targets, nb)
        else:
            tg = targets      # 消融：故意不同步 GT 框（错误做法，见 A12 记录 3.3 节）
        _, g = _one_grad(dets[name], xt, tg, cfg.get("seed", 0) + t)
        v = mu * v + l1_norm(g)
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def rntme_nae(dets: Dict[str, Detector], x, targets, cfg) -> torch.Tensor:
    """RNTME-NAE 检测版：随机变换集成 + 动态权重 + 高斯平滑 + Nesterov。

    cfg: eps alpha T mu B tau beta sigma gamma_geo gamma_col rel_norm
    """
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    B = cfg.get("B", 5)
    tau, beta = cfg.get("tau", 2.0), cfg.get("beta", 0.7)
    sigma, xi = cfg.get("sigma", 3.0), 1e-6
    g_geo, g_col = cfg.get("gamma_geo", 1.0), cfg.get("gamma_col", 3.0)
    rel = cfg.get("rel_norm", True)
    names: List[str] = list(dets.keys())
    M = len(names)
    dev = x.device
    H, W = x.shape[-2], x.shape[-1]

    x_adv = x.clone()
    v = torch.zeros_like(x)
    w_bar = torch.full((M,), 1.0 / M, device=dev)
    base = torch.ones(M, device=dev)          # 各模型首步损失，用于相对归一化
    gen = torch.Generator(device=dev).manual_seed(cfg.get("seed", 0))

    for t in range(T):
        p_geo = math.exp(-g_geo * t / max(T, 1))
        p_col = math.exp(-g_col * t / max(T, 1))
        s_lo = 0.80 + 0.20 * p_geo
        br = ct = 0.25 * p_col
        bl = min(3.0 * p_col, 0.02 * min(H, W))

        x_nes = x_adv + mu * v
        # —— 逐副本变换：偶数副本走几何（框同步），奇数副本走颜色
        imgs, tgs = [], []
        for b in range(B):
            if b % 2 == 0:
                # 注意：这里不能用变量名 xi —— 上面的 xi 是权重下限保护常量，
                # 一旦被图像张量覆盖，后面 (Lin/tau + xi) 会广播成一整张图。
                xgi, s, tx, ty = rand_geo(x_nes, s_lo, 1.0, 0.06, gen)
                imgs.append(xgi)
                if cfg.get("warp_boxes", True):
                    nb = [warp_boxes(tt["boxes"], s[i], tx[i], ty[i], W, H)
                          for i, tt in enumerate(targets)]
                    tgs.append(_filter_targets(targets, nb))
                else:
                    tgs.append(targets)   # 消融：几何变换后仍用原框
            else:
                imgs.append(rand_color(x_nes, br, ct, bl, gen))
                tgs.append([{"boxes": tt["boxes"], "labels": tt["labels"]} for tt in targets])
        # 同一模型在各副本上分别求梯度，再按副本平均
        losses, grads = [], []
        for nm in names:
            gi = torch.zeros_like(x_adv)
            li = torch.zeros(x_adv.shape[0], device=dev)
            for b in range(B):
                L, g = _one_grad(dets[nm], imgs[b], tgs[b], cfg.get("seed", 0) + 100 * t + b)
                gi = gi + g
                li = li + L
            g = smooth(gi / B, sigma)
            losses.append(li / B)
            grads.append(l1_norm(g))
        L = torch.stack(losses)                      # (M, n)
        if t == 0:
            base = L.mean(1).detach().clamp_min(1e-6)
        Lin = L / base.unsqueeze(1) if rel else L
        r = (Lin / tau + xi).reciprocal()
        w = r / r.sum(0, keepdim=True)
        w_bar = beta * w_bar + (1 - beta) * w.mean(1)
        w_bar = w_bar / w_bar.sum()
        g = torch.zeros_like(x_adv)
        for i in range(M):
            g = g + w_bar[i] * grads[i]
        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


REGISTRY = {
    "I-FGSM": ifgsm,
    "NAE": nae,
    "DI-NI-FGSM": di_ni_fgsm,
    "RNTME-NAE": rntme_nae,
}


def run(name: str, dets: Dict[str, Detector], x, targets, cfg) -> torch.Tensor:
    return REGISTRY[name](dets, x, targets, cfg)
