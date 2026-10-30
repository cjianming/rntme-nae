# -*- coding: utf-8 -*-
"""A3 主实验：对抗样本 × 防御 × 模型的完整评估矩阵。

三阶段，可单跑也可连跑：

    # 1) 生成对抗样本（复用 A1.1 的攻击实现与同配置）
    python -m a13.run --stage gen --n 600 --T 15 --B 5

    # 2) 自适应攻击（BPDA）：攻击者知道防御存在，前向过真防御、反向当恒等
    python -m a13.run --stage gen --bpda jpeg75,bit4,blur10 --attacks RNTME-NAE

    # 3) 全部防御 × 全部攻击 × 全部模型
    python -m a13.run --stage eval

指标口径与 A1.1 完全一致（ASR = 全部样本上的误分类比例），另加：

    ASR_rel = (acc_clean_def - acc_adv_def) / acc_clean_def
        「扣掉防御自身准确率损失后」的相对攻击率。randpad 这类防御
        本身会掉点，只看绝对 ASR 会把它的代价算到攻击头上。

    net_gain = (ASR_none - ASR_def) - (acc_clean_none - acc_clean_def)
        净防御收益：ASR 降了多少，减去干净准确率掉了多少。
        这个数 ≤ 0 的防御，本质上只是把模型弄笨。
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A          # noqa: E402
from a11 import data as DA           # noqa: E402
from a11 import models as M          # noqa: E402
from a11 import swasr as S           # noqa: E402
from a13 import defenses as DF       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
CKPT = os.path.join(ROOT, "a11", "chk")


def load(arch, size, device):
    p = os.path.join(CKPT, f"{arch}_s{size}.pt")
    ck = torch.load(p, map_location=device, weights_only=False)
    m = M.build(arch, 43, size).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


@torch.no_grad()
def predict_def(m, x, dname, tries=5):
    """在某个防御下做预测；randpad 用 tries 次随机采样的多数投票。"""
    if dname == "randpad":
        votes = [m(DF.rand_pad(x, 0.85).clamp(-1, 1)).argmax(1) for _ in range(tries)]
        v = torch.stack(votes)                       # (tries, B)
        out = []
        for i in range(v.shape[1]):
            vals, cnt = torch.unique(v[:, i], return_counts=True)
            out.append(vals[cnt.argmax()])
        return torch.stack(out)
    xd = DF.DEFENSES[dname](x).clamp(-1, 1)
    return m(xd).argmax(1)


def get_data(size, n, bs):
    ext = DA.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    _, te = DA._read_index(ext)
    te = DA.stratified_subset(te, n, seed=2)
    ds = DA.GTSRB(te, size, False, cache=True)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0), len(te)


# ------------------------------------------------------------------ 阶段 1


def stage_gen(args, models, ens, device):
    """生成并保存对抗样本。每攻击一个 npz。"""
    loader, n = get_data(args.size, args.n, args.bs)
    cfg = dict(eps=args.eps, alpha=args.alpha, T=args.T, mu=args.mu,
               source=args.source, B=args.B, tau=args.tau, beta=args.beta,
               sigma=args.sigma, early_stop=False, seed=0, win=5, rho=0.9, delta=0.05)
    atks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    bpdas = [b.strip() for b in args.bpda.split(",") if b.strip()] if args.bpda else [""]

    for bp in bpdas:
        tag = f"{args.out}_{'bpda_' + bp if bp else 'plain'}"
        for atk in atks:
            path = os.path.join(RES, f"{tag}_{atk}.npz")
            if args.reuse and os.path.exists(path):
                print(f"  跳过（已有）{os.path.basename(path)}")
                continue
            # 攻击源模型：BPDA 时把防御套在外面（反向当恒等）
            if bp:
                src = DF.build(bp, ens, bpda=True)
            else:
                src = ens
            if atk in ("I-FGSM", "NAE", "DI-NI-FGSM"):
                one = src[args.source]
                src_use = {args.source: one}
            else:
                src_use = src
            t0 = time.time()
            adv, ys, l2s, cps = [], [], [], []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xa = A.run(atk, src_use, x, y, cfg)
                adv.append(DF.to_uint8(xa))
                ys.append(y.cpu().numpy().astype(np.int64))
                l2s.append((xa - x).flatten(1).norm(dim=1).cpu().numpy())
                cps.append(models[args.source](x).argmax(1).cpu().numpy())
            np.savez_compressed(
                path,
                x_adv=np.concatenate(adv),
                y=np.concatenate(ys),
                l2=np.concatenate(l2s).astype(np.float32),
                clean_pred=np.concatenate(cps),
            )
            dt = time.time() - t0
            print(f"  [gen] {tag:<16s}{atk:<12s}{n} 张 {dt:6.1f}s "
                  f"({dt/n*1000:5.1f} ms/张) -> {os.path.basename(path)}", flush=True)


# ------------------------------------------------------------------ 阶段 2


def stage_eval(args, models, ens_names, device):
    loader, n = get_data(args.size, args.n, args.bs)
    W = S.build_matrix()

    # 先把干净图像缓存成一张张量，供各防御复用
    xs = torch.cat([x for x, _ in loader]).to(device)
    ys = torch.cat([y for _, y in loader]).to(device)

    dlist = [d.strip() for d in args.defenses.split(",") if d.strip()]
    atks = [a.strip() for a in args.attacks.split(",") if a.strip()]

    # 干净样本在各防御下的准确率（防御的代价）
    clean_def = {}
    for d in dlist:
        for nm, m in models.items():
            acc = (predict_def(m, xs, d) == ys).float().mean().item()
            clean_def[(d, nm)] = acc
        print(f"  [clean] {d:<10s} " +
              " ".join(f"{nm}={clean_def[(d, nm)]:.3f}" for nm in models), flush=True)

    rows = []
    # 对照：无防御下的对抗样本（沿用 plain npz）
    for atk in atks:
        for d in dlist:
            path = os.path.join(RES, f"{args.out}_plain_{atk}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺 {path}，先跑 --stage gen")
            z = np.load(path)
            xa = DF.from_uint8(z["x_adv"], device)
            yy = z["y"]
            for nm, m in models.items():
                pr = predict_def(m, xa, d).cpu().numpy()
                met = S.metrics_from_preds(yy, pr, W)
                acc_adv = 1.0 - met["ASR"]
                c0 = clean_def[(d, nm)]
                rows.append({
                    "kind": "普通攻击", "defense": d, "attack": atk, "target": nm,
                    "role": "集成内" if nm in ens_names else "未见",
                    "clean_acc": c0, "acc_adv": acc_adv,
                    "ASR": met["ASR"], "SW-ASR": met["SW-ASR"], "DMR": met["DMR"],
                    "ASR_rel": (c0 - acc_adv) / max(c0, 1e-6),
                    "L2": float(z["l2"].mean()),
                })

    # 自适应攻击（BPDA）
    if args.bpda:
        for bp in [b.strip() for b in args.bpda.split(",") if b.strip()]:
            for atk in atks:
                path = os.path.join(RES, f"{args.out}_bpda_{bp}_{atk}.npz")
                if not os.path.exists(path):
                    continue
                z = np.load(path)
                xa = DF.from_uint8(z["x_adv"], device)
                yy = z["y"]
                for nm, m in models.items():
                    pr = predict_def(m, xa, bp).cpu().numpy()
                    met = S.metrics_from_preds(yy, pr, W)
                    acc_adv = 1.0 - met["ASR"]
                    c0 = clean_def[(bp, nm)]
                    rows.append({
                        "kind": f"自适应({bp})", "defense": bp, "attack": atk,
                        "target": nm, "role": "集成内" if nm in ens_names else "未见",
                        "clean_acc": c0, "acc_adv": acc_adv,
                        "ASR": met["ASR"], "SW-ASR": met["SW-ASR"], "DMR": met["DMR"],
                        "ASR_rel": (c0 - acc_adv) / max(c0, 1e-6),
                        "L2": float(z["l2"].mean()),
                    })

    # 净防御收益（相对于 none 那一行）
    base = {}
    for r in rows:
        if r["kind"] == "普通攻击" and r["defense"] == "none":
            base[(r["attack"], r["target"])] = r
    for r in rows:
        b = base.get((r["attack"], r["target"]))
        if b:
            r["net_gain"] = (b["ASR"] - r["ASR"]) - (b["clean_acc"] - r["clean_acc"])
        else:
            r["net_gain"] = float("nan")

    csv_path = os.path.join(RES, f"{args.out}_defense.csv")
    cols = ["kind", "defense", "attack", "target", "role", "clean_acc", "acc_adv",
            "ASR", "SW-ASR", "DMR", "ASR_rel", "net_gain", "L2"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, ""))
                        for k in cols})

    # 汇总打印：按防御 × 攻击，集成内平均 / 未见
    print("\n== 汇总（ASR，括号内为 ASR_rel）==")
    print(f"{'防御':<10s}{'攻击':<13s}{'集成内':>16s}{'未见':>16s}")
    for d in dlist:
        for atk in atks:
            v1 = [r["ASR"] for r in rows if r["defense"] == d and r["attack"] == atk
                  and r["kind"] == "普通攻击" and r["role"] == "集成内"]
            v2 = [r["ASR"] for r in rows if r["defense"] == d and r["attack"] == atk
                  and r["kind"] == "普通攻击" and r["role"] == "未见"]
            r1 = [r["ASR_rel"] for r in rows if r["defense"] == d and r["attack"] == atk
                  and r["kind"] == "普通攻击" and r["role"] == "集成内"]
            r2 = [r["ASR_rel"] for r in rows if r["defense"] == d and r["attack"] == atk
                  and r["kind"] == "普通攻击" and r["role"] == "未见"]
            if v1:
                print(f"{d:<10s}{atk:<13s}{np.mean(v1):.3f}({np.mean(r1):.3f})"
                      f"{np.mean(v2) if v2 else float('nan'):>10.3f}"
                      f"({np.mean(r2) if r2 else float('nan'):.3f})")

    meta = dict(n=n, size=args.size, T=args.T, B=args.B, eps=args.eps,
                ensemble=[e for e in args.ensemble.split(",") if e],
                heldout=[h for h in args.heldout.split(",") if h],
                defenses=dlist, attacks=atks, bpda=args.bpda,
                clean_acc={f"{k[0]}|{k[1]}": v for k, v in clean_def.items()},
                weight_version=S.VERSION if hasattr(S, "VERSION") else "v2")
    with open(os.path.join(RES, f"{args.out}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\n写出 {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["gen", "eval", "all"])
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--eps", type=float, default=16 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--mu", type=float, default=0.9)
    p.add_argument("--B", type=int, default=5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--ensemble", default="sresnet,svgg,ssep")
    p.add_argument("--heldout", default="sinc")
    p.add_argument("--source", default="sresnet")
    p.add_argument("--attacks", default="I-FGSM,NAE,DI-NI-FGSM,RNTME-NAE")
    p.add_argument("--defenses", default="none,jpeg75,jpeg50,jpeg420,bit4,bit3,blur10,randpad,gamma06,dehaze")
    p.add_argument("--bpda", default="", help="对这些防御做自适应攻击，逗号分隔")
    p.add_argument("--reuse", action="store_true", help="已有 npz 则跳过生成")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", default="a3")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(RES, exist_ok=True)
    device = "cpu"
    ens_names = [s.strip() for s in args.ensemble.split(",") if s.strip()]
    hold_names = [s.strip() for s in args.heldout.split(",") if s.strip()]

    print(f"设备 {device} | 集成 {ens_names} | 未见 {hold_names} | 输入 {args.size}px")
    models = {nm: load(nm, args.size, device) for nm in ens_names + hold_names}
    ens = {k: models[k] for k in ens_names}
    print("  模型 " + " ".join(
        f"{k}={M.describe(v)/1e6:.2f}M" for k, v in models.items()))

    if args.stage in ("gen", "all"):
        stage_gen(args, models, ens, device)
    if args.stage in ("eval", "all"):
        stage_eval(args, models, ens_names, device)


if __name__ == "__main__":
    main()
