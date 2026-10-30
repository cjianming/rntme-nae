# -*- coding: utf-8 -*-
"""攻击实现：I-FGSM / NAE / DI-NI-FGSM / RNTME-NAE。

统一接口：
    x_adv = run(name, models, x, y, cfg)
其中 models 是 {名字: 已 eval() 的 nn.Module}，返回的 x_adv 与 x 同形状、已裁到 eps 球内。

所有实现都按 batch 计算；早停以 batch 为单位（一旦该 batch 内所有样本对全部模型均已攻破即停），
这是工程近似，论文里需注明「早停粒度 = batch」。
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


# ------------------------------------------------------------ 基础算子
def _grad(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return g.detach()


def _l1_norm(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """逐样本 L1 归一化（MI-FGSM / NAE 的标准做法）。"""
    d = g.flatten(1).abs().sum(1).clamp_min(eps)
    return g / d.view(-1, *([1] * (g.dim() - 1)))


def _gauss_kernel(sigma: float, k: int | None = None) -> torch.Tensor:
    """可分离高斯核，用于梯度低通平滑（对应论文 3.5 节）。"""
    if k is None:
        k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    xs = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def smooth(g: torch.Tensor, sigma: float) -> torch.Tensor:
    """对梯度做高斯平滑：先横向再纵向，逐通道卷积。"""
    if sigma <= 0:
        return g
    k = _gauss_kernel(sigma)
    ks = k.numel()
    c = g.shape[1]
    kh = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    kw = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
    pad = ks // 2
    g = F.conv2d(F.pad(g, (pad, pad, 0, 0), mode="reflect"), kh, groups=c)
    g = F.conv2d(F.pad(g, (0, 0, pad, pad), mode="reflect"), kw, groups=c)
    return g


# ------------------------------------------------------------ 随机变换
def rand_transform(x: torch.Tensor, scale_lo: float, scale_hi: float,
                   bright: float, contrast: float, blur_sigma: float,
                   gen: torch.Generator | None = None) -> torch.Tensor:
    """几何（随机缩放 + 平移 + 镜像补边）+ 颜色（亮度/对比度）+ 模糊。

    对应论文 3.3 节六类变换的简化组合。几何部分用单次 grid_sample 批量完成，
    逐样本独立采样参数，等价于 B 个副本各自变换。
    """
    n = x.shape[0]
    dev, dt = x.device, x.dtype
    out = x

    # 1) 几何：每个样本一个缩放系数 + 一个小平移，越界处镜像填充
    if scale_hi > scale_lo:
        s = torch.empty(n, device=dev, dtype=dt).uniform_(scale_lo, scale_hi, generator=gen)
        t = (0.06 * (1.0 - s.clamp(0, 1))).unsqueeze(1)
        tx = torch.empty(n, 1, device=dev, dtype=dt).uniform_(-1, 1, generator=gen) * t
        ty = torch.empty(n, 1, device=dev, dtype=dt).uniform_(-1, 1, generator=gen) * t
        inv = (1.0 / s).unsqueeze(1)
        zero = torch.zeros(n, 1, device=dev, dtype=dt)
        theta = torch.cat([inv, zero, tx, zero, inv, ty], dim=1).view(n, 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="reflection",
                            align_corners=False)

    # 2) 亮度
    if bright > 0:
        f = torch.empty(n, 1, 1, 1, device=dev, dtype=dt).uniform_(1 - bright, 1 + bright, generator=gen)
        out = out * f
    # 3) 对比度（向自身均值收缩）
    if contrast > 0:
        mu = out.mean(dim=(1, 2, 3), keepdim=True)
        f = torch.empty(n, 1, 1, 1, device=dev, dtype=dt).uniform_(1 - contrast, 1 + contrast, generator=gen)
        out = (out - mu) * f + mu
    # 4) 高斯模糊（模拟失焦）
    if blur_sigma > 0:
        k = _gauss_kernel(blur_sigma, 5).to(dev, dt)
        ks = k.numel()
        c = out.shape[1]
        kh = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
        kw = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
        p = ks // 2
        out = F.conv2d(F.pad(out, (p, p, 0, 0), mode="reflect"), kh, groups=c)
        out = F.conv2d(F.pad(out, (0, 0, p, p), mode="reflect"), kw, groups=c)
    return out.clamp(-1, 1)


# ------------------------------------------------------------ 各攻击
@torch.no_grad()
def _finish(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x_adv, x - eps, x + eps).clamp(-1, 1)


def ifgsm(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    eps, alpha, T = cfg["eps"], cfg["alpha"], cfg["T"]
    name = cfg.get("source", next(iter(models)))
    x_adv = x.clone()
    for _ in range(T):
        g = _grad(models[name], x_adv, y)
        x_adv = _finish(x_adv + alpha * g.sign(), x, eps)
    return x_adv.detach()


def mi_fgsm(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """MI-FGSM（Dong et al. 2018）：动量累积 + L1 归一化梯度，单模型。

    修回补充：主表原先只列 I-FGSM 而未列 MI-FGSM，无法区分「动量」与「本文机制」的贡献，
    故在统一框架内补实现（与 I-FGSM 同一源模型、同一步长与迭代预算）。
    """
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source", next(iter(models)))
    m = models[name]
    x_adv = x.clone()
    v = torch.zeros_like(x)
    for _ in range(T):
        g = _l1_norm(_grad(m, x_adv, y))
        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def nae(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """Nesterov 前瞻动量 + L1 归一化梯度。"""
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source", next(iter(models)))
    m = models[name]
    x_adv = x.clone()
    v = torch.zeros_like(x)
    for _ in range(T):
        x_nes = x_adv + mu * v
        g = _l1_norm(_grad(m, x_nes, y))
        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def di_ni_fgsm(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """DI + NI：输入多样性（随机缩放补边）+ Nesterov，单模型。"""
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source", next(iter(models)))
    m = models[name]
    x_adv = x.clone()
    v = torch.zeros_like(x)
    gen = torch.Generator(device=x.device).manual_seed(cfg.get("seed", 0))
    for _ in range(T):
        x_nes = x_adv + mu * v
        xt = rand_transform(x_nes, 0.85, 1.0, 0.0, 0.0, 0.0, gen)
        g = _l1_norm(_grad(m, xt, y))
        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def rntme_nae(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """RNTME-NAE：随机变换集成 (RT) + 动态权重集成 (DME) + 高斯平滑 (TI) + Nesterov。

    cfg 关键项：
      B        每个样本的变换副本数（表 3 取 5）
      M        参与集成的替代模型数（= len(models)，表 3 取 5）
      tau      权重温度（2.0）
      beta     权重 EMA 平滑系数（0.7）
      sigma    高斯核参数（3）
      gamma_geo / gamma_col  几何、颜色支路的衰减速率（1.0 / 3.0）
      xi       权重下限保护（1e-6）
      early_stop  是否启用滑动窗口早停
      win, rho, delta  早停窗口长度、攻破阈值、波动阈值
    """
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    B = cfg.get("B", 5)
    tau, beta = cfg.get("tau", 2.0), cfg.get("beta", 0.7)
    sigma, xi = cfg.get("sigma", 3), cfg.get("xi", 1e-6)
    g_geo, g_col = cfg.get("gamma_geo", 1.0), cfg.get("gamma_col", 3.0)
    names: List[str] = list(models.keys())
    M = len(names)
    dev = x.device

    x_adv = x.clone()
    v = torch.zeros_like(x)
    w_bar = torch.full((M,), 1.0 / M, device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.get("seed", 0))

    win = cfg.get("win", 5)
    rho, delta = cfg.get("rho", 0.9), cfg.get("delta", 0.05)
    use_es = cfg.get("early_stop", False)
    hist = torch.zeros(M, win, dtype=torch.bool, device=dev)
    filled = 0

    for t in range(T):
        # —— 自适应强度：几何缓降、颜色速降（式(6)）
        p_geo = math.exp(-g_geo * t / max(T, 1))
        p_col = math.exp(-g_col * t / max(T, 1))
        s_lo = 0.75 + 0.25 * p_geo          # 缩放下界 1.0 -> 0.75
        br = 0.25 * p_col                   # 亮度幅度 0.25 -> 0
        ct = 0.25 * p_col
        # 模糊 sigma 上限随分辨率缩放：32 px 取到 1.9 已接近全糊，224 px 才用满 3.6
        bl = min(3.6 * p_col, 0.06 * x.shape[-1])

        x_nes = x_adv + mu * v
        # B 个变换副本（前若干个走几何变换，其余走颜色变换，两类交替）
        reps = []
        for b in range(B):
            if b % 2 == 0:
                reps.append(rand_transform(x_nes, 0.75 + 0.25 * p_geo, 1.0,
                                           0.0, 0.0, 0.0, gen))
            else:
                reps.append(rand_transform(x_nes, 1.0, 1.0, br, ct, bl, gen))
        xr = torch.cat(reps, 0)                       # (B*n, C, H, W)
        yr = y.repeat(B)

        # —— 各模型的损失与梯度
        losses, grads, hits = [], [], []
        for nm in names:
            xi_ = xr.detach().requires_grad_(True)
            logits = models[nm](xi_)
            loss = F.cross_entropy(logits, yr, reduction="none").view(B, -1).mean(1)
            g, = torch.autograd.grad(loss.sum(), xi_)
            # 副本梯度求平均后平滑
            g = g.detach().view(B, -1, *g.shape[1:]).mean(0)
            g = smooth(g, sigma)
            losses.append(loss.detach())
            grads.append(_l1_norm(g))
            with torch.no_grad():
                hits.append((logits.argmax(1) != yr).view(B, -1).float().mean(1) >= rho)
        L = torch.stack(losses)                        # (M, n)
        if cfg.get("equal_weight", False):
            # 消融对照：与完整模型唯一的差别是权重恒为 1/M（式(7)—(9) 整段旁路），
            # 随机变换、高斯平滑、Nesterov 三条支路完全相同。
            w_bar = torch.full((M,), 1.0 / M, device=dev)
        else:
            # —— 损失倒数加权（式(7)(8)）
            r = (L / tau + xi).reciprocal()
            w = r / r.sum(0, keepdim=True)
            # —— EMA 平滑（式(9)）
            w_bar = beta * w_bar + (1 - beta) * w.mean(1)
            w_bar = w_bar / w_bar.sum()
        # —— 融合梯度（式(10)）
        g = torch.zeros_like(x_adv)
        for i in range(M):
            g = g + w_bar[i] * grads[i]

        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)

        # —— 滑动窗口早停（式(15)）
        if use_es:
            cur = torch.stack(hits).all(0)             # (n,) 该样本对所有模型均攻破
            hist = torch.roll(hist, -1, dims=1)
            hist[:, -1] = cur
            filled = min(filled + 1, win)
            if filled >= win:
                if hist.all():
                    break
    return x_adv.detach()


def admix_mi(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """Admix（Wu et al., ICCV 2021）+ MI-FGSM 动量：把输入与「其他类别的图像」线性混合，
    对 m1 个混合副本取平均梯度，以此扩宽输入分布。

    cfg 关键项：
      m1   混合副本数（原文 5）
      eta  混合比例（原文 0.2）
      admix_pool  混合源图像池 [K,3,H,W]；缺省时用 batch 内 roll 兜底
    """
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    m1 = int(cfg.get("m1", 5))
    eta = float(cfg.get("eta", 0.2))
    m = models[cfg.get("source", next(iter(models)))]
    pool = cfg.get("admix_pool", None)
    gen = torch.Generator(device=x.device).manual_seed(int(cfg.get("seed", 0)) + 977)

    x_adv = x.clone()
    v = torch.zeros_like(x)
    for _ in range(T):
        g_sum = torch.zeros_like(x)
        for k in range(m1):
            if pool is not None and len(pool) > 1:
                idx = torch.randint(0, len(pool), (x.shape[0],), generator=gen)
                xp = pool[idx].to(x.device)
            else:
                xp = x.roll(k + 1, dims=0)
            xm = torch.clamp(x_adv + eta * xp, -1.0, 1.0)
            g_sum = g_sum + _grad(m, xm, y)
        g = _l1_norm(g_sum / m1)
        v = mu * v + g
        x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    return x_adv.detach()


def _sgm_handles(model: torch.nn.Module, gamma: float):
    """SGM（Wu et al., ICLR 2021）：对残差块「主分支」末端卷积回传的梯度乘 gamma，
    使梯度更多经由 skip connection 传播。对无残差结构的模型自动不生效。"""
    hs = []
    for mod in model.modules():
        if type(mod).__name__ == "_Basic":
            hs.append(mod.c2.register_full_backward_pre_hook(
                lambda _m, go, _g=gamma: tuple(t * _g for t in go)))
    return hs


def sgm_mi(models: Dict[str, torch.nn.Module], x, y, cfg) -> torch.Tensor:
    """MI-FGSM + SGM 梯度衰减：结构与 MI-FGSM 唯一差别是残差主分支的梯度被衰减。"""
    gamma = float(cfg.get("gamma", 0.5))
    eps, alpha, T, mu = cfg["eps"], cfg["alpha"], cfg["T"], cfg["mu"]
    name = cfg.get("source", next(iter(models)))
    m = models[name]
    hs = _sgm_handles(m, gamma)
    try:
        x_adv = x.clone()
        v = torch.zeros_like(x)
        for _ in range(T):
            g = _l1_norm(_grad(m, x_adv, y))
            v = mu * v + g
            x_adv = _finish(x_adv + alpha * v.sign(), x, eps)
    finally:
        for h in hs:
            h.remove()
    return x_adv.detach()


ATTACKS = {
    "I-FGSM": ifgsm,
    "MI-FGSM": mi_fgsm,
    "NAE": nae,
    "DI-NI-FGSM": di_ni_fgsm,
    "ADMIX": admix_mi,
    "SGM": sgm_mi,
    "RNTME-NAE": rntme_nae,
}

# 使用集成的方法（其余只用源模型，保证「模型数量」这一因素被显式控制）
ENSEMBLE_ATTACKS = {"RNTME-NAE"}


def run(name: str, models, x, y, cfg):
    """models 传已满集；内部按方法类型裁剪为「单源」或「集成」。equal_weight 走 cfg。"""
    if name.startswith("RNTME-EQ"):
        cfg = dict(cfg, equal_weight=True)
        name = "RNTME-NAE"
    elif name.startswith("RNTME"):
        cfg = dict(cfg, equal_weight=False)
        name = "RNTME-NAE"
    src = models if name in ENSEMBLE_ATTACKS else {cfg.get("source", next(iter(models))):
                                                   models[cfg.get("source", next(iter(models)))]}
    return ATTACKS[name](src, x, y, cfg)
