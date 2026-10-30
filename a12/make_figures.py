# -*- coding: utf-8 -*-
"""A1.2 图表：攻击前后检测框对比 + mAP 下降柱状图。"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

_FONT = None
for cand in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
             r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(cand):
        _FONT = font_manager.FontProperties(fname=cand)
        break
plt.rcParams["axes.unicode_minus"] = False

C_GREEN = "#2f7d4f"
C_RED = "#c0392b"
C_GREY = "#8a8a8a"


def load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def draw(ax, img_np, det, title, color, thr=0.3):
    ax.imshow(img_np)
    n = 0
    for b, s, l in zip(det["boxes"], det["scores"], det["labels"]):
        if s < thr:
            continue
        n += 1
        x0, y0, x1, y1 = b
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=color, lw=1.8))
        ax.text(x0, max(y0 - 4, 8), f"{int(l)}:{s:.2f}", color="white", fontsize=6,
                bbox=dict(facecolor=color, edgecolor="none", pad=1.0))
    ax.set_title(f"{title}（检出 {n}）", fontproperties=_FONT, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_examples(tag: str, detector: str, attack: str, n_show: int = 3,
                 n_img: int = 80, data_seed: int = 0, size=(384, 512)):
    from .train import SynthSet
    ds = SynthSet("test", n_img, size=size, seed=data_seed + 7777)
    clean = load_npz(os.path.join(RES, f"{tag}_clean.npz"))
    adv = load_npz(os.path.join(RES, f"{tag}_{attack}.npz"))

    cb = clean[f"clean_{detector}_boxes"]
    cs = clean[f"clean_{detector}_scores"]
    cl = clean[f"clean_{detector}_labels"]
    ab = adv[f"{attack}_{detector}_boxes"]
    as_ = adv[f"{attack}_{detector}_scores"]
    al = adv[f"{attack}_{detector}_labels"]

    # 选「干净检出多、攻击后检出少」的样本
    gain = []
    for i in range(len(ds)):
        c = int(((cs[i] >= 0.3)).sum())
        a = int(((as_[i] >= 0.3)).sum())
        gain.append((c - a, c, i))
    gain.sort(reverse=True)
    idx = [g[2] for g in gain[:n_show]]

    fig, axes = plt.subplots(n_show, 2, figsize=(7.2, 2.55 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 2)
    for r, i in enumerate(idx):
        arr = np.asarray(ds[i]["img"].permute(1, 2, 0).numpy())
        draw(axes[r, 0], arr, {"boxes": cb[i], "scores": cs[i], "labels": cl[i]},
             f"原图 #{i}　干净", C_GREEN)
        draw(axes[r, 1], arr, {"boxes": ab[i], "scores": as_[i], "labels": al[i]},
             f"对抗样本（{attack}）", C_RED)
    fig.tight_layout(pad=0.6)
    out = os.path.join(FIG, "figA_det_compare.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_drop(tag: str):
    with open(os.path.join(RES, f"{tag}_summary.json"), encoding="utf-8") as f:
        J = json.load(f)
    rows = J["rows"]
    attacks = list(dict.fromkeys(r["attack"] for r in rows))
    roles = ["集成内", "未见"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(attacks))
    w = 0.36
    for k, role in enumerate(roles):
        vals = []
        for a in attacks:
            sub = [r["drop_pct"] for r in rows if r["attack"] == a and r["role"] == role]
            vals.append(np.mean(sub) if sub else np.nan)
        axes[0].bar(x + (k - 0.5) * w, vals, w,
                    label=role, color=C_GREEN if role == "集成内" else C_GREY)
        for xi, v in zip(x + (k - 0.5) * w, vals):
            axes[0].text(xi, v + 0.8, f"{v:.1f}", ha="center", fontproperties=_FONT, fontsize=7)

    for k, role in enumerate(roles):
        vals = []
        for a in attacks:
            sub = [r["dMR"] for r in rows if r["attack"] == a and r["role"] == role]
            vals.append(np.mean(sub) if sub else np.nan)
        axes[1].bar(x + (k - 0.5) * w, vals, w,
                    label=role, color=C_GREEN if role == "集成内" else C_GREY)
        for xi, v in zip(x + (k - 0.5) * w, vals):
            axes[1].text(xi, v + 0.004, f"{v*100:.1f}", ha="center",
                         fontproperties=_FONT, fontsize=7)

    for ax, ttl, ylab in [(axes[0], "mAP@0.5 相对下降", "下降 / %"),
                          (axes[1], "单类漏检率增量", "ΔMR")]:
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, fontproperties=_FONT, fontsize=8)
        ax.set_title(ttl, fontproperties=_FONT, fontsize=10)
        ax.set_ylabel(ylab, fontproperties=_FONT, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.legend(prop=_FONT, fontsize=7.5, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG, "figB_drop.png")
    fig.savefig(out, dpi=420, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--detector", default="ssdlite_a")
    ap.add_argument("--attack", default="RNTME-NAE")
    args = ap.parse_args()
    print(fig_examples(args.tag, args.detector, args.attack))
    print(fig_drop(args.tag))
