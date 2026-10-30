# -*- coding: utf-8 -*-
"""出图：单栏 8.6 cm，600 dpi，中文用黑体。

figA  SW-ASR-DMR by attack    ：四个攻击的三指标对照，说明 ASR 掩盖了严重度差异
figB  top confusion pairs     ：攻击实际把哪些标志改成了什么，按语义权重上色

用法：python -m a11.make_figures --prefix pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import swasr as S        # noqa: E402
from a11.analyze import CN        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
HEI = "C:/Windows/Fonts/simhei.ttf"
CM = 1 / 2.54


def FP(s):
    return FontProperties(fname=HEI, size=s)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "savefig.dpi": 600,
})

C_BLUE, C_ORANGE, C_RED = "#3d6f9e", "#d08a3e", "#a83a32"
C_GREY = "#8a8a8a"


def load_summary(prefix):
    p = os.path.join(RES, f"{prefix}_summary.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_a(summ, prefix):
    rows = summ["summary"]
    names = [r["attack"] for r in rows]
    asr = [r["in_ASR"] * 100 for r in rows]
    sw = [r["in_SW"] * 100 for r in rows]
    dmr = [r["in_DMR"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8.6 * CM, 6.4 * CM))
    n = len(names)
    idx = np.arange(n)
    w = 0.26
    kw = dict(edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(idx - w, asr, w, color=C_BLUE, label="ASR", **kw)
    ax.bar(idx, sw, w, color=C_ORANGE, label="SW-ASR", **kw)
    ax.bar(idx + w, dmr, w, color=C_RED, label="DMR", **kw)
    for xs, vs in ((idx - w, asr), (idx, sw), (idx + w, dmr)):
        for x, v in zip(xs, vs):
            ax.text(x, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontproperties=FP(5.4))
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontproperties=FP(6.0))
    ax.set_ylabel("百分比 / %", fontproperties=FP(6.4))
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=5.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, prop=FP(5.8), loc="upper left", ncol=3,
              handlelength=0.9, columnspacing=0.9, handletextpad=0.4,
              bbox_to_anchor=(0, 1.02))
    ax.set_title("(a) 集成内目标上的三项指标", fontproperties=FP(6.8), pad=4)
    fig.tight_layout(pad=0.35)
    out = os.path.join(FIG, "figA_metrics.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("->", out)


def fig_b(prefix, atk="RNTME-NAE", top=8):
    files = glob.glob(os.path.join(RES, f"{prefix}_{atk}_*.npz"))
    if not files:
        print("缺少 npz，跳过 figB")
        return
    W = S.build_matrix()
    merged = Counter()
    cls = Counter()
    for f in files:
        d = np.load(f)
        for t, p in zip(d["true"], d["pred"]):
            cls[int(t)] += 1
            if t != p:
                merged[(int(t), int(p))] += 1
    pairs = merged.most_common(top)
    if not pairs:
        print("没有误分类，跳过 figB")
        return

    labels = [f"{CN[t]}→{CN[p]}" for (t, p), _ in pairs]
    vals = [c for _, c in pairs]
    ws = [W[t][p] for (t, p), _ in pairs]
    colors = [C_RED if w >= 0.8 else (C_ORANGE if w >= 0.5 else C_GREY) for w in ws]

    fig, ax = plt.subplots(figsize=(8.6 * CM, 6.2 * CM))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    for yy, v, w in zip(y, vals, ws):
        ax.text(v + max(vals) * 0.015, yy, f"{v}  (w={w:.1f})", va="center",
                fontproperties=FP(5.2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=FP(5.8))
    ax.set_xlabel("误分类样本数", fontproperties=FP(6.4))
    ax.set_xlim(0, max(vals) * 1.32)
    ax.tick_params(axis="x", labelsize=5.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"(b) {atk} 造成的主要混淆对（红=严重放宽，橙=明显放宽，灰=其他）",
                 fontproperties=FP(6.6), pad=4)
    fig.tight_layout(pad=0.35)
    out = os.path.join(FIG, "figB_confusions.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("->", out)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot")
    args = ap.parse_args()
    summ = load_summary(args.prefix)
    fig_a(summ, args.prefix)
    fig_b(args.prefix)


if __name__ == "__main__":
    main()
