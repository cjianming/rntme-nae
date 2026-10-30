# -*- coding: utf-8 -*-
"""修回补充实验 R1：多种子 + 未见目标 + 统一框架的多方法比较。

对应大修意见条件二与条件三：
  · 同一批方法（含 MI-FGSM 与等权集成，补齐主表缺失的两个对照）
  · 目标在生成阶段完全不可见（--heldout 从不进入 --ensemble）
  · 不少于 5 个种子，给出均值±标准差
  · Wilcoxon 符号秩检验 + Bonferroni 校正

用法：
    python -m a11.run_seeds --size 64 --n 300 --seeds 0,1,2,3,4 \
        --ensemble sresnet,svgg,ssep --heldout sinc --out r1

    python -m a11.run_seeds --mode scan --n 120 --tscan 5,8,11,15   # 迭代预算扫描
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A      # noqa: E402
from a11 import data as D         # noqa: E402
from a11 import models as M       # noqa: E402
from a11 import swasr as S        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "chk")
RES = os.path.join(HERE, "results")

DEFAULT_ATTACKS = ["I-FGSM", "MI-FGSM", "NAE", "DI-NI-FGSM", "RNTME-EQ", "RNTME-NAE"]

# 只在这些攻击里，随机种子才会改变结果（其余为确定性攻击，多次运行结果相同）
STOCHASTIC = {"DI-NI-FGSM", "RNTME-EQ", "RNTME-NAE"}


def load(arch, size, device="cpu"):
    p = os.path.join(CKPT, f"{arch}_s{size}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 {p}，先运行 train.py --arch {arch} --size {size}")
    ck = torch.load(p, map_location=device, weights_only=False)
    m = M.build(arch, 43, size).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, float(ck.get("test_acc", float("nan"))), float(ck.get("val_best", float("nan")))


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


def build_subset(n, size, seed=2):
    ext = D.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    _, test_items = D._read_index(ext)
    return D.stratified_subset(test_items, n, seed=seed)


def make_cfg(args, seed, pool=None):
    cfg = dict(eps=args.eps255 / 255.0, alpha=args.alpha255 / 255.0, T=args.T,
               mu=args.mu, source=args.source, B=args.B, tau=args.tau,
               beta=args.beta, sigma=args.sigma, early_stop=False,
               seed=seed, win=5, rho=0.9, delta=0.05)
    if pool is not None:
        cfg["admix_pool"] = pool
    return cfg


def build_pool(n, size, seed=7):
    """Admix 的混合源：从**训练**集分层抽样，与评测子集不相交（评测取测试集）。"""
    ext = D.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    train_items, _ = D._read_index(ext)
    items = D.stratified_subset(train_items, n, seed=seed)
    ld = DataLoader(D.GTSRB(items, size, False, cache=True),
                    batch_size=64, shuffle=False, num_workers=0)
    return torch.cat([x for x, _ in ld], 0)


def metrics(true, pred, clean_pred, W):
    m = S.metrics_from_preds(true, pred, W)
    ok = clean_pred == true
    m["ASR_cond"] = float((pred[ok] != true[ok]).mean()) if ok.any() else float("nan")
    m["n_cond"] = int(ok.sum())
    return m


def run_seeds(args):
    device = "cpu"
    ens_names = [s.strip() for s in args.ensemble.split(",") if s.strip()]
    hold_names = [s.strip() for s in args.heldout.split(",") if s.strip()]
    all_names = ens_names + [h for h in hold_names if h not in ens_names]
    args.source = args.source or ens_names[0]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    atks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    os.makedirs(RES, exist_ok=True)
    W = S.build_matrix()

    print(f"设备 {device} | 集成 {ens_names} | 未见 {hold_names} | 源 {args.source}")
    print(f"种子 {seeds} | 方法 {atks} | 样本 {args.n} | 尺寸 {args.size}px")

    models, acc = {}, {}
    for nm in all_names:
        models[nm], acc[nm], _ = load(nm, args.size, device)
        print(f"  {nm:<10s} 存档测试准确率 {acc[nm]:.4f}")
    ens = {k: models[k] for k in ens_names}

    items = build_subset(args.n, args.size, seed=2)
    loader = DataLoader(D.GTSRB(items, args.size, False, cache=True),
                        batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"评测样本 {len(items)}（分层抽样，seed=2，所有种子共用同一批）")
    pool = None
    if any(a.startswith("ADMIX") for a in atks):
        pool = build_pool(300, args.size, seed=7)
        print(f"Admix 混合池 {tuple(pool.shape)}（训练集分层抽样，seed=7，与评测集不相交）")

    # 干净预测（一次算好，用于条件 ASR）
    with torch.no_grad():
        clean_pred, ys, xs = {}, [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        X = torch.cat(xs, 0)
        Y = torch.cat(ys, 0)
        for nm, m in models.items():
            clean_pred[nm] = predict(m, X).cpu().numpy()
    Yn = Y.cpu().numpy()
    clean_acc = {nm: float((clean_pred[nm] == Yn).mean()) for nm in models}
    print("  本子集干净准确率: " + "  ".join(f"{k}={v:.4f}" for k, v in clean_acc.items()))

    n = len(Yn)
    rec = {(a, sd): {} for a in atks for sd in seeds}
    timing = {}
    for a in atks:
        for sd in seeds:
            t0 = time.time()
            preds = {nm: [] for nm in models}
            for i in range(0, n, args.bs):
                x = X[i:i + args.bs]
                y = Y[i:i + args.bs]
                cfg = make_cfg(args, sd, pool)
                xa = A.run(a, ens, x, y, cfg)
                for nm, m in models.items():
                    preds[nm].append(predict(m, xa).cpu().numpy())
            dt = time.time() - t0
            timing[(a, sd)] = dt / n
            for nm in models:
                rec[(a, sd)][nm] = np.concatenate(preds[nm])
            print(f"  [{a:<10s} seed={sd}] {dt:.1f}s（{dt/n*1000:.0f} ms/样本）", flush=True)

    # ---------- 聚合：每 (方法, 目标) 跨种子的均值±标准差 ----------
    keys = ["ASR", "SW-ASR", "DMR", "ASR_cond"]
    agg = {}
    for a in atks:
        for nm in models:
            per = []
            for sd in seeds:
                m = metrics(Yn, rec[(a, sd)][nm], clean_pred[nm], W)
                per.append(m)
            agg[(a, nm)] = {k: (float(np.mean([p[k] for p in per])),
                                float(np.std([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0)
                            for k in keys}
            agg[(a, nm)]["ms"] = float(np.mean([timing[(a, sd)] for sd in seeds]) * 1000)

    role = {nm: ("集成内" if nm in ens_names else "未见") for nm in models}
    rows = []
    for a in atks:
        for nm in models:
            r = {"attack": a, "target": nm, "role": role[nm], "clean_acc": clean_acc[nm]}
            for k in keys:
                mu, sd = agg[(a, nm)][k]
                r[k] = mu
                r[k + "_std"] = sd
            r["ms"] = agg[(a, nm)]["ms"]
            rows.append(r)
    csv_path = os.path.join(RES, f"{args.out}_seeds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\n== 跨种子均值±标准差（{len(seeds)} 种子）==")
    hdr = f"{'方法':<11s}{'目标':<9s}{'角色':<5s}{'ASR':>16s}{'条件ASR':>16s}{'SW-ASR':>16s}{'DMR':>16s}{'ms':>7s}"
    print(hdr)
    for a in atks:
        for nm in models:
            g = agg[(a, nm)]
            def fm(k):
                mu, sd = g[k]
                return f"{mu*100:.1f}±{sd*100:.1f}"
            print(f"{a:<11s}{nm:<9s}{role[nm]:<5s}{fm('ASR'):>16s}{fm('ASR_cond'):>16s}"
                  f"{fm('SW-ASR'):>16s}{fm('DMR'):>16s}{g['ms']:>7.0f}")
        print("-" * len(hdr))

    # ---------- 显著性检验 ----------
    tests = []
    ref = "RNTME-NAE" if "RNTME-NAE" in atks else atks[0]
    for nm in models:
        base_val = {a: np.mean([(rec[(a, sd)][nm] != Yn).astype(float) for sd in seeds], 0)
                    for a in atks}
        base_sw = {a: np.mean([np.array([W[t][p] for t, p in zip(Yn, rec[(a, sd)][nm])])
                               for sd in seeds], 0) for a in atks}
        others = [a for a in atks if a != ref]
        m = len(others) * len(models) * 2        # 比较数：目标 × 对照方法 × 指标(ASR/SW)
        for a in others:
            for tag, d1, d0 in (("ASR", base_val[ref], base_val[a]),
                                ("SW-ASR", base_sw[ref], base_sw[a])):
                diff = d1 - d0
                if np.allclose(diff, 0):
                    p = 1.0
                    stat = 0.0
                else:
                    stat, p = wilcoxon(d1, d0, zero_method="wilcox",
                                       alternative="two-sided",
                                       correction=False, method="approx")
                tests.append({"target": nm, "role": role[nm], "metric": tag,
                              "ref": ref, "other": a,
                              "delta_pp": float(diff.mean() * 100),
                              "W": float(stat), "p_raw": float(p),
                              "p_bonf": float(min(1.0, p * m)),
                              "signif_bonf": bool(p * m < 0.05),
                              "n_pairs": int(len(diff))})
    print(f"\n== Wilcoxon 符号秩检验（配对单位 = 样本 n={n}，跨种子取均值；Bonferroni m={m}）==")
    for t in tests:
        flag = "显著" if t["signif_bonf"] else "不显著"
        print(f"  {t['target']:<9s} {t['metric']:<7s} {ref} vs {t['other']:<11s} "
              f"Δ={t['delta_pp']:+6.2f}pp  p={t['p_raw']:.3g}  p_bonf={t['p_bonf']:.3g}  {flag}")

    with open(os.path.join(RES, f"{args.out}_time.json"), "w", encoding="utf-8") as f:
        json.dump({f"{a}|{sd}": timing[(a, sd)] for a in atks for sd in seeds},
                  f, ensure_ascii=False, indent=1)

    json_path = os.path.join(RES, f"{args.out}_seeds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "clean_acc": clean_acc,
                   "n_samples": n, "seeds": seeds, "attacks": atks,
                   "ensemble": ens_names, "heldout": hold_names,
                   "rows": rows, "tests": tests},
                  f, ensure_ascii=False, indent=2, default=str)

    # 逐样本存档（供后续重算指标）
    np.savez_compressed(os.path.join(RES, f"{args.out}_raw.npz"),
                        **{f"{a}|{sd}|{nm}": rec[(a, sd)][nm]
                           for a in atks for sd in seeds for nm in models},
                        true=Yn,
                        **{f"clean|{nm}": clean_pred[nm] for nm in models})
    print(f"\n明细 {csv_path}\n汇总 {json_path}")


def run_scan(args):
    """迭代预算扫描：回答「第 8 轮已贴住 ε 球，第 9—15 轮还有没有收益」。（意见 M8）"""
    device = "cpu"
    ens_names = [s.strip() for s in args.ensemble.split(",") if s.strip()]
    hold_names = [s.strip() for s in args.heldout.split(",") if s.strip()]
    args.source = args.source or ens_names[0]
    models, acc = {}, {}
    for nm in ens_names + hold_names:
        models[nm], acc[nm], _ = load(nm, args.size, device)
    ens = {k: models[k] for k in ens_names}
    items = build_subset(args.n, args.size, seed=2)
    loader = DataLoader(D.GTSRB(items, args.size, False, cache=True),
                        batch_size=args.bs, shuffle=False)
    X = torch.cat([x for x, _ in loader], 0)
    Y = torch.cat([y for _, y in loader], 0)
    Yn = Y.numpy()
    Ts = [int(t) for t in args.tscan.split(",")]
    atks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    out = []
    print(f"迭代扫描 T={Ts} | 样本 {len(Yn)} | 尺寸 {args.size}px")
    for a in atks:
        for T in Ts:
            t0 = time.time()
            preds = {nm: [] for nm in models}
            l2s, maxstep = [], []
            for i in range(0, len(Yn), args.bs):
                x, y = X[i:i + args.bs], Y[i:i + args.bs]
                cfg = make_cfg(args, seed=0)
                cfg["T"] = T
                xa = A.run(a, ens, x, y, cfg)
                l2s.append((xa - x).flatten(1).norm(dim=1).cpu().numpy())
                maxstep.append((xa - x).abs().flatten(1).max(dim=1).values.cpu().numpy())
                for nm, m in models.items():
                    preds[nm].append(predict(m, xa).cpu().numpy())
            dt = time.time() - t0
            row = {"attack": a, "T": T, "ms": dt / len(Yn) * 1000,
                   "L2": float(np.concatenate(l2s).mean()),
                   "sat_frac": float((np.concatenate(maxstep) >= args.eps255 / 255.0 - 1e-3).mean())}
            for nm in models:
                pr = np.concatenate(preds[nm])
                row[f"ASR_{nm}"] = float((pr != Yn).mean())
            out.append(row)
            print(f"  {a:<11s} T={T:<3d} 饱和比例 {row['sat_frac']:.3f}  L2 {row['L2']:.2f}  "
                  + "  ".join(f"{nm}={row[f'ASR_{nm}']:.3f}" for nm in models)
                  + f"  {row['ms']:.0f}ms")
    p = os.path.join(RES, f"{args.out}_tscan.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # CSV
    cp = os.path.join(RES, f"{args.out}_tscan.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n{cp}\n{p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="seeds", choices=["seeds", "scan"])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--eps255", type=int, default=16)
    p.add_argument("--alpha255", type=int, default=2)
    p.add_argument("--mu", type=float, default=0.9)
    p.add_argument("--B", type=int, default=5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--ensemble", default="sresnet,svgg")
    p.add_argument("--heldout", default="ssep")
    p.add_argument("--source", default="")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    p.add_argument("--tscan", default="5,8,11,15")
    p.add_argument("--out", default="r1")
    args = p.parse_args()
    (run_seeds if args.mode == "seeds" else run_scan)(args)


if __name__ == "__main__":
    main()
