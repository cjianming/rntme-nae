# -*- coding: utf-8 -*-
"""A3 出图：单栏 8.6 cm / 通栏 17 cm，600 dpi，中文黑体。

fig1  pareto   ：防御的「代价—收益」散点，横轴干净准确率损失，纵轴 ASR
fig2  heat     ：防御 × 攻击的 ASR 热力图（集成内 / 未见 两栏）
fig3  bpda     ：普通攻击 vs 自适应攻击，看防御是否被绕过

用法：python -m a13.make_figures --out a3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a13.defenses import DESC      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
HEI = "C:/Windows/Fonts/simhei.ttf"
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


def FP(s):
    return FontProperties(fname=HEI, size=s)


C_IN = "#1f4e79"      # 集成内
C_UN = "#c0504d"      # 未见
GRID = dict(color="#d9d9d9", lw=0.5)


def load(out):
    p = os.path.join(RES, f"{out}_defense.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        for k in ("clean_acc", "acc_adv", "ASR", "SW-ASR", "DMR", "ASR_rel", "net_gain", "L2"):
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = float("nan")
    return rows


def agg(rows, d, atk=None, kind="普通攻击", role=None):
    sub = [r for r in rows if r["defense"] == d and r["kind"] == kind]
    if atk:
        sub = [r for r in sub if r["attack"] == atk]
    if role:
        sub = [r for r in sub if r["role"] == role]
    v = [r["ASR"] for r in sub]
    return float(np.mean(v)) if v else float("nan")


def agg_clean(rows, d, role=None):
    sub = [r for r in rows if r["defense"] == d and r["kind"] == "普通攻击"]
    if role:
        sub = [r for r in sub if r["role"] == role]
    v = [r["clean_acc"] for r in sub]
    return float(np.mean(v)) if v else float("nan")


# ------------------------------------------------------------------ fig1


def fig_pareto(out):
    rows = load(out)
    dlist = [d for d in DESC if any(r["defense"] == d for r in rows)]
    base_c = agg_clean(rows, "none")
    # 先算出全部坐标，统一 y 轴范围，避免 sharey 把高点裁掉
    allpts = {}
    for role in ("集成内", "未见"):
        allpts[role] = sorted(
            [((base_c - agg_clean(rows, d, role)) * 100, agg(rows, d, role=role) * 100, d)
             for d in dlist], key=lambda t: (t[0], t[1]))
    ys_all = [p[1] for v in allpts.values() for p in v]
    ylo, yhi = min(ys_all) - 3.5, max(ys_all) + 3.5

    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.6 * CM))
    for ax, role, col, ttl in ((axes[0], "集成内", C_IN, "(a) 集成内目标"),
                               (axes[1], "未见", C_UN, "(b) 未见目标")):
        pts = allpts[role]
        seen = []                              # 已标注的 x，用于垂直堆叠
        for x, y, d in pts:
            big = d == "none"
            ax.scatter(x, y, s=54 if big else 30, c=col, edgecolor="white",
                       linewidth=0.8, zorder=3, marker="*" if big else "o")
            k = sum(1 for px in seen if abs(px - x) < 0.40)   # 同 x 邻域的已用层数
            dy = 4.5 + 7.2 * (k % 3)
            ax.annotate(d, (x, y), textcoords="offset points", xytext=(5.0, dy),
                        fontproperties=FP(5.6),
                        color="#111111" if big else "#4a4a4a")
            seen.append(x)
        ax.set_xlabel("干净准确率损失 / pp", fontproperties=FP(7.0))
        ax.set_title(ttl, fontproperties=FP(7.4), pad=3.5)
        ax.grid(True, **GRID)
        ax.tick_params(labelsize=6.0, length=2.2)
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            lb.set_fontproperties(FP(6.2))
        xs = [p[0] for p in pts]
        ax.set_xlim(min(-0.6, min(xs) - 0.7), max(xs) + 1.5)
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("四攻击平均 ASR / %", fontproperties=FP(7.0))
    fig.tight_layout(pad=0.35)
    p = os.path.join(FIG, "fig1_pareto.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------------ fig2


def fig_heat(out):
    rows = load(out)
    dlist = [d for d in DESC if any(r["defense"] == d for r in rows)]
    atks = []
    for r in rows:
        if r["kind"] == "普通攻击" and r["attack"] not in atks:
            atks.append(r["attack"])
    atks = sorted(atks, key=lambda a: ["I-FGSM", "NAE", "DI-NI-FGSM", "RNTME-NAE"]
                  .index(a) if a in ["I-FGSM", "NAE", "DI-NI-FGSM", "RNTME-NAE"] else 9)

    cmap = LinearSegmentedColormap.from_list(
        "wr", ["#f7f7f7", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"])
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7.6 * CM))
    for ax, role, ttl in ((axes[0], "集成内", "(a) 集成内目标"),
                          (axes[1], "未见", "(b) 未见目标")):
        M = np.array([[agg(rows, d, a, role=role) * 100 for a in atks] for d in dlist])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(atks)))
        ax.set_xticklabels(atks, fontproperties=FP(6.0), rotation=18, ha="right")
        if role == "集成内":
            ax.set_yticks(range(len(dlist)))
            ax.set_yticklabels(dlist, fontproperties=FP(6.4))
        else:
            ax.set_yticks([])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontproperties=FP(5.8),
                        color="white" if v > 62 else "#222222")
        ax.set_title(ttl, fontproperties=FP(7.4), pad=3.5)
        ax.set_xticks(np.arange(-.5, len(atks), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(dlist), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.1)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label("ASR / %", fontproperties=FP(6.6))
    cb.ax.tick_params(labelsize=5.8)
    fig.savefig(os.path.join(FIG, "fig2_heat.png"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(FIG, "fig2_heat.png")


# ------------------------------------------------------------------ fig3


def fig_bpda(out):
    rows = load(out)
    kinds = [k for k in dict.fromkeys(r["kind"] for r in rows) if k != "普通攻击"]
    if not kinds:
        return None
    defs = [k.replace("自适应(", "").replace(")", "") for k in kinds]
    atks = [a for a in dict.fromkeys(r["attack"] for r in rows if r["kind"] != "普通攻击")]
    atk = "RNTME-NAE" if "RNTME-NAE" in atks else atks[0]

    fig, ax = plt.subplots(figsize=(10.0 * CM, 5.6 * CM))
    x = np.arange(len(defs))
    w = 0.34
    v0 = [agg(rows, d, atk, role="集成内") * 100 for d in defs]
    v1 = [agg(rows, d, atk, kind=f"自适应({d})", role="集成内") * 100 for d in defs]
    base = agg(rows, "none", atk, role="集成内") * 100
    b0 = ax.bar(x - w / 2, v0, w, color="#a6a6a6", edgecolor="white", lw=0.6,
                label="普通攻击")
    b1 = ax.bar(x + w / 2, v1, w, color=C_UN, edgecolor="white", lw=0.6,
                label="自适应攻击（BPDA）")
    ax.axhline(base, color="#1f4e79", lw=0.9, ls="--")
    ax.text(len(defs) - 0.42, base + 1.8, f"无防御 {base:.1f}%", ha="right",
            fontproperties=FP(5.8), color="#1f4e79")
    for bs, vs in ((b0, v0), (b1, v1)):
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width() / 2, v - 5.5, f"{v:.1f}",
                    ha="center", fontproperties=FP(5.6), color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([DESC[d][0] for d in defs], fontproperties=FP(6.2))
    ax.set_ylabel(f"{atk} 的 ASR / %", fontproperties=FP(7.0))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.0, length=2.2)
    for lb in ax.get_yticklabels():
        lb.set_fontproperties(FP(6.2))
    ax.legend(prop=FP(6.0), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(pad=0.3)
    p = os.path.join(FIG, "fig3_bpda.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="a3")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for fn in (fig_pareto, fig_heat, fig_bpda):
        p = fn(a.out)
        print("  ->", p)


if __name__ == "__main__":
    main()
