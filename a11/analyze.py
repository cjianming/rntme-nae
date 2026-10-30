# -*- coding: utf-8 -*-
"""把逐样本预测变成「攻击到底改成了什么」的清单。

对交通读者来说，比 ASR 更有信息量的是：
    扰动把「限速 30」变成了「限速 80」共 47 次，占该类被攻击样本的 53%，语义权重 1.0。
这一句同时给出了频率、方向与后果。

用法：
    python -m a11.analyze --prefix pilot --top 12
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import swasr as S   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CN = {
    0: "限速20", 1: "限速30", 2: "限速50", 3: "限速60", 4: "限速70", 5: "限速80",
    6: "解除限速80", 7: "限速100", 8: "限速120", 9: "禁止超车",
    10: "禁止3.5t以上超车", 11: "前方路口优先", 12: "干道优先", 13: "让行",
    14: "停车让行", 15: "禁止机动车通行", 16: "禁止3.5t以上车辆", 17: "禁止驶入",
    18: "一般危险警告", 19: "左急弯", 20: "右急弯", 21: "连续弯道", 22: "路面不平",
    23: "易滑路面", 24: "右侧变窄", 25: "前方施工", 26: "交通信号灯", 27: "行人",
    28: "儿童", 29: "自行车", 30: "冰雪", 31: "野生动物", 32: "解除全部限制",
    33: "右转", 34: "左转", 35: "直行", 36: "直行或右转", 37: "直行或左转",
    38: "靠右", 39: "靠左", 40: "环岛", 41: "解除禁止超车", 42: "解除3.5t禁止超车",
}

CONSEQ = {
    1.0: "显著放宽，可直接导致违章或事故",
    0.8: "明显放宽，需驾驶员纠正",
    0.5: "同族混淆，风险量级相当",
    0.3: "更严格，属安全侧错误",
}


def fmt(v: float) -> str:
    return "—" if v is None else f"{v:.2f}"


def load_npz(pattern: str):
    """返回 {attack: {target: (true, pred)}}"""
    out = {}
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)[:-4]
        d = np.load(f)
        out[base] = (d["true"], d["pred"])
    return out


def top_pairs(true, pred, W, k=12):
    cnt = Counter()
    per_cls = Counter()
    for t, p in zip(true, pred):
        per_cls[int(t)] += 1
        if t != p:
            cnt[(int(t), int(p))] += 1
    rows = []
    for (t, p), c in cnt.most_common(k):
        rows.append({
            "true": t, "pred": p, "count": c,
            "share": c / max(per_cls[t], 1),
            "w": W[t][p],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--weights", default="")
    args = ap.parse_args()

    W = S.load_csv(args.weights) if args.weights and os.path.exists(args.weights) else S.build_matrix()
    files = load_npz(os.path.join(RES, f"{args.prefix}_*.npz"))
    if not files:
        print(f"{RES} 下没有 {args.prefix}_*.npz，先跑 eval_attack.py")
        return

    summary = {}
    for key, (true, pred) in files.items():
        # 文件名形如 pilot_RNTME-NAE_svgg
        parts = key[len(args.prefix) + 1:].rsplit("_", 1)
        atk, tgt = parts[0], parts[1]
        rows = top_pairs(true, pred, W, args.top)
        summary.setdefault(atk, {})[tgt] = rows

    for atk in summary:
        print(f"\n{'='*78}\n攻击：{atk}")
        # 在所有目标上合并统计，得到最稳定的混淆对
        merged = Counter()
        cls_cnt = Counter()
        for tgt, rows in summary[atk].items():
            for r in rows:
                merged[(r["true"], r["pred"])] += r["count"]
                cls_cnt[r["true"]] += r["count"]
        print(f"{'真实标志':<16s}{'被判为':<16s}{'次数':>6s}{'权重':>6s}  交通后果")
        print("-" * 78)
        for (t, p), c in merged.most_common(args.top):
            w = W[t][p]
            tag = "危险" if w >= 0.8 else ("同族" if w >= 0.5 else "安全侧")
            print(f"{CN[t]:<16s}{CN[p]:<16s}{c:>6d}{w:>6.2f}  [{tag}] {CONSEQ.get(round(w,1), '')}")

    out = os.path.join(RES, f"{args.prefix}_confusions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n明细 {out}")


if __name__ == "__main__":
    main()
