# -*- coding: utf-8 -*-
"""用 v2 权重表重算试点指标。

v1 -> v2 的改动是「安全侧错误不再记 0」：把限速 80 读成停车标志会让车辆
在高速路段幽灵急停，不能说它无害。这一步只影响 SW-ASR 的绝对值，
DMR 用 w>=0.8 阈值，低风险档改动不跨阈值，故 DMR 不变。

用法：python -m a11.recompute_v2 --prefix pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

import numpy as np

from . import swasr as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot")
    args = ap.parse_args()

    csv_path = os.path.join(RES, f"{args.prefix}.csv")
    json_path = os.path.join(RES, f"{args.prefix}_summary.json")

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with open(json_path, encoding="utf-8") as f:
        J = json.load(f)

    W = S.build_matrix()
    old = {(r["attack"], r["target"]): r for r in rows}

    # 逐 (攻击, 目标) 从 npz 重算
    changed = 0
    for r in rows:
        npz = os.path.join(RES, f"{args.prefix}_{r['attack']}_{r['target']}.npz")
        d = np.load(npz)
        met = S.metrics_from_preds(d["true"], d["pred"], W)
        prev = float(r["SW-ASR"])
        r["ASR"] = f"{met['ASR']:.4f}"
        r["SW-ASR"] = f"{met['SW-ASR']:.4f}"
        r["DMR"] = f"{met['DMR']:.4f}"
        if abs(met["SW-ASR"] - prev) > 1e-4:
            changed += 1

    # 备份旧版
    if not os.path.exists(os.path.join(RES, f"{args.prefix}_v1.csv")):
        shutil.copy(csv_path, os.path.join(RES, f"{args.prefix}_v1.csv"))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 重算汇总
    summ = []
    for atk in dict.fromkeys(r["attack"] for r in rows):
        sub = [r for r in rows if r["attack"] == atk]
        ins = [r for r in sub if r["role"] == "集成内"]
        out = [r for r in sub if r["role"] == "未见"]
        mean = lambda rs, k: sum(float(r[k]) for r in rs) / len(rs) if rs else float("nan")
        summ.append({
            "attack": atk,
            "in_ASR": mean(ins, "ASR"), "in_SW": mean(ins, "SW-ASR"), "in_DMR": mean(ins, "DMR"),
            "out_ASR": mean(out, "ASR"), "out_SW": mean(out, "SW-ASR"),
            "L2": mean(sub, "L2"), "SSIM": mean(sub, "SSIM"),
            "ms": mean(sub, "sec_per_sample") * 1000,
        })

    J["weight_version"] = "v2"
    J["summary"] = summ
    J["rows"] = [{
        "attack": r["attack"], "target": r["target"], "role": r["role"],
        "clean_acc": float(r["clean_acc"]), "ASR": float(r["ASR"]),
        "SW-ASR": float(r["SW-ASR"]), "DMR": float(r["DMR"]),
        "L2": float(r["L2"]), "SSIM": float(r["SSIM"]),
        "sec_per_sample": float(r["sec_per_sample"]),
    } for r in rows]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=2, default=str)

    print(f"{'攻击':<12s}{'集成内 ASR':>11s}{'SW-ASR v2':>11s}{'DMR':>8s}{'未见 ASR':>10s}{'未见 SW':>9s}")
    for s in summ:
        print(f"{s['attack']:<12s}{s['in_ASR']:>11.4f}{s['in_SW']:>11.4f}{s['in_DMR']:>8.4f}"
              f"{s['out_ASR']:>10.4f}{s['out_SW']:>9.4f}")

    print(f"\n重算 {len(rows)} 行，SW-ASR 变化 {changed} 行（其余为 0 权重未被触发，数值不变）")
    print(f"CSV  {csv_path}\nJSON {json_path}\n旧版 {args.prefix}_v1.csv")


if __name__ == "__main__":
    main()
