# -*- coding: utf-8 -*-
"""合并多种子结果：均值±标准差 + Wilcoxon 符号秩检验（Bonferroni 校正）。

输入：若干 {prefix}_raw.npz（由 run_seeds.py 分种子并行产出，键格式 "攻击|种子|目标"）
输出：{out}_seeds.csv、{out}_tests.csv、{out}_summary.json

用法：
    python -m a11.merge_seeds --inputs r1s0,r1s1,r1s2,r1s3,r1s4 --out r1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import swasr as S        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

KEYS = ["ASR", "SW-ASR", "DMR", "ASR_cond"]


def load_all(prefixes):
    """返回 {seed: {attack: {target: pred}}}、true、clean。"""
    data, true, clean = {}, None, {}
    for pfx in prefixes:
        p = os.path.join(RES, f"{pfx}_raw.npz")
        z = np.load(p)
        per = {}
        for k in z.files:
            if k.startswith("clean|"):
                clean[k.split("|", 1)[1]] = z[k]
                continue
            if k == "true":
                if true is None:
                    true = z[k]
                continue
            a, sd, nm = k.split("|")
            per.setdefault(int(sd), {}).setdefault(a, {})[nm] = z[k]
        for sd, d in per.items():
            data.setdefault(sd, {}).update(d)
    return data, true, clean


def metrics(true, pred, clean_pred, W):
    m = S.metrics_from_preds(true, pred, W)
    ok = clean_pred == true
    m["ASR_cond"] = float((pred[ok] != true[ok]).mean()) if ok.any() else float("nan")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="r1s0,r1s1,r1s2,r1s3,r1s4")
    p.add_argument("--out", default="r1")
    p.add_argument("--ref", default="RNTME-NAE")
    p.add_argument("--heldout", default="sinc", help="不参与生成的目标名，仅用于标注 role")
    args = p.parse_args()

    prefixes = [s.strip() for s in args.inputs.split(",") if s.strip()]
    data, true, clean = load_all(prefixes)
    seeds = sorted(data.keys())
    W = S.build_matrix()
    atks = sorted({a for sd in seeds for a in data[sd]})
    targets = sorted({nm for sd in seeds for a in data[sd] for nm in data[sd][a]})
    print(f"合并 {len(prefixes)} 个文件 | 种子 {seeds} | 方法 {atks} | 目标 {targets} | 样本 {len(true)}")

    rows = []
    for a in atks:
        for nm in targets:
            per = [metrics(true, data[sd][a][nm], clean[nm], W) for sd in seeds]
            r = {"attack": a, "target": nm, "n_seeds": len(seeds),
                 "role": ("未见" if nm in args.heldout.split(",") else "集成内")}
            for k in KEYS:
                v = np.array([q[k] for q in per], dtype=float)
                r[k] = float(v.mean())
                r[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rows.append(r)

    cp = os.path.join(RES, f"{args.out}_seeds.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\n== 跨 {len(seeds)} 种子的均值±标准差（单位 %）==")
    hdr = (f"{'方法':<11s}{'目标':<9s}{'ASR':>15s}{'条件ASR':>15s}{'SW-ASR':>15s}"
           f"{'DMR':>15s}")
    print(hdr)
    for a in atks:
        for nm in targets:
            r = next(x for x in rows if x["attack"] == a and x["target"] == nm)
            def fm(k):
                return f"{r[k]*100:.1f}±{r[k+'_std']*100:.1f}"
            print(f"{a:<11s}{nm:<9s}{fm('ASR'):>15s}{fm('ASR_cond'):>15s}"
                  f"{fm('SW-ASR'):>15s}{fm('DMR'):>15s}")
        print("-" * len(hdr))

    # ---------------- 显著性检验 ----------------
    def per_sample(a, nm, kind):
        """逐样本指标，跨种子取均值：ASR 用指示值，SW-ASR 用权重值。"""
        vals = []
        for sd in seeds:
            pr = data[sd][a][nm]
            if kind == "ASR":
                vals.append((pr != true).astype(float))
            else:
                vals.append(np.array([W[int(t)][int(q)] for t, q in zip(true, pr)], dtype=float))
        return np.mean(vals, 0)

    others = [a for a in atks if a != args.ref]
    n_test = len(targets) * len(others) * 2
    tests = []
    for nm in targets:
        for a in others:
            for kind in ("ASR", "SW-ASR"):
                d1 = per_sample(args.ref, nm, kind)
                d0 = per_sample(a, nm, kind)
                diff = d1 - d0
                if np.allclose(diff, 0):
                    stat, praw = 0.0, 1.0
                else:
                    stat, praw = wilcoxon(d1, d0, zero_method="wilcox",
                                          alternative="two-sided",
                                          correction=False, method="approx")
                tests.append({"target": nm, "metric": kind, "ref": args.ref, "other": a,
                              "delta_pp": float(diff.mean() * 100),
                              "W": float(stat), "p_raw": float(praw),
                              "p_bonf": float(min(1.0, praw * n_test)),
                              "signif": bool(praw * n_test < 0.05),
                              "n_pairs": int(len(diff)),
                              "n_nonzero": int(np.count_nonzero(diff))})
    print(f"\n== Wilcoxon 符号秩检验（配对单位 = 样本 n={len(true)}，跨种子取均值；"
          f"Bonferroni m={n_test}）==")
    for t in tests:
        print(f"  {t['target']:<9s} {t['metric']:<7s} {args.ref} vs {t['other']:<11s} "
              f"Δ={t['delta_pp']:+6.2f}pp  p={t['p_raw']:.3g}  p_bonf={t['p_bonf']:.3g}  "
              f"{'显著' if t['signif'] else '不显著'}")

    tp = os.path.join(RES, f"{args.out}_tests.csv")
    with open(tp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        w.writeheader()
        for r in tests:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    jp = os.path.join(RES, f"{args.out}_summary.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"inputs": prefixes, "seeds": seeds, "attacks": atks,
                   "heldout": args.heldout.split(","),
                   "targets": targets, "n_samples": int(len(true)),
                   "n_tests": n_test, "rows": rows, "tests": tests},
                  f, ensure_ascii=False, indent=2)
    print(f"\n{cp}\n{tp}\n{jp}")


if __name__ == "__main__":
    main()
