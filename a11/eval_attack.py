# -*- coding: utf-8 -*-
"""跑 A1.1：在 GTSRB 上生成对抗样本，按「集成内目标 / 未见目标」分别统计。

用法（CPU 试点）：
    python -m a11.eval_attack --size 32 --n 600 --T 15 \
        --ensemble sresnet,svgg,ssep --heldout sinc --out pilot

用法（GPU 全量）：
    python -m a11.eval_attack --size 224 --n 5000 --T 15 \
        --ensemble resnet50,efficientnet_b0,vgg19,mobilenet_v2,googlenet \
        --heldout resnet101,densenet121 --out full
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A     # noqa: E402
from a11 import data as D        # noqa: E402
from a11 import models as M      # noqa: E402
from a11 import swasr as S       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "chk")
RES = os.path.join(HERE, "results")


def load(arch, size, device):
    p = os.path.join(CKPT, f"{arch}_s{size}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 {p}，先运行 train.py --arch {arch}")
    ck = torch.load(p, map_location=device, weights_only=False)
    m = M.build(arch, 43, size).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, ck.get("test_acc", float("nan"))


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


def ssim_mean(a, b):
    """全局 SSIM（高斯窗 11，sigma 1.5），返回 batch 均值。"""
    c = a.shape[1]
    xs = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    k = torch.exp(-(xs ** 2) / (2 * 1.5 ** 2))
    k = (k / k.sum())
    kh = k.view(1, 1, 1, 11).repeat(c, 1, 1, 1)
    kw = k.view(1, 1, 11, 1).repeat(c, 1, 1, 1)
    pad = lambda z: (F.pad(z, (5, 5, 0, 0), mode="reflect"), F.pad(z, (0, 0, 5, 5), mode="reflect"))
    def blur(z):
        z = F.conv2d(pad(z)[0], kh, groups=c)
        return F.conv2d(pad(z)[1], kw, groups=c)
    mu1, mu2 = blur(a), blur(b)
    s1 = blur(a * a) - mu1 ** 2
    s2 = blur(b * b) - mu2 ** 2
    s12 = blur(a * b) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    v = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return v.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--n", type=int, default=600, help="测试样本数")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--eps", type=float, default=16 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--mu", type=float, default=0.9)
    p.add_argument("--B", type=int, default=5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--sigma", type=float, default=1.5, help="32px 试点用小核，全量取 3")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--ensemble", default="sresnet,svgg,ssep")
    p.add_argument("--heldout", default="sinc")
    p.add_argument("--source", default="")
    p.add_argument("--attacks", default="I-FGSM,NAE,DI-NI-FGSM,RNTME-NAE")
    p.add_argument("--weights", default="", help="可选：自定义 SW-ASR 权重 CSV")
    p.add_argument("--out", default="pilot")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ens_names = [s.strip() for s in args.ensemble.split(",") if s.strip()]
    hold_names = [s.strip() for s in args.heldout.split(",") if s.strip()]
    source = args.source or ens_names[0]

    os.makedirs(RES, exist_ok=True)
    W = S.load_csv(args.weights) if args.weights and os.path.exists(args.weights) else S.build_matrix()

    print(f"设备 {device} | 集成 {ens_names} | 未见 {hold_names} | 源 {source}")
    models, clean = {}, {}
    for nm in ens_names + [h for h in hold_names if h not in ens_names]:
        m, acc = load(nm, args.size, device)
        models[nm] = m
        clean[nm] = acc
        print(f"  {nm:<16s} 测试准确率(存档) {acc:.4f}")
    ens = {k: models[k] for k in ens_names}

    # 数据
    ext = D.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    _, test_items = D._read_index(ext)
    test_items = D.stratified_subset(test_items, args.n, seed=2)
    loader = DataLoader(D.GTSRB(test_items, args.size, False, cache=True), batch_size=args.bs,
                        shuffle=False, num_workers=0)
    print(f"评测样本 {len(test_items)}")

    # 各目标上的干净准确率（用本次评测子集重算，与存档值可能微差）
    with torch.no_grad():
        for nm, m in models.items():
            c = n = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                c += (predict(m, x) == y).sum().item()
                n += y.numel()
            clean[nm] = c / n
    print("  本子集干净准确率: " + "  ".join(f"{k}={v:.4f}" for k, v in clean.items()))

    cfg = dict(eps=args.eps, alpha=args.alpha, T=args.T, mu=args.mu, source=source,
               B=args.B, tau=args.tau, beta=args.beta, sigma=args.sigma,
               early_stop=args.early_stop, seed=0, win=5, rho=0.9, delta=0.05)

    rows = []
    for atk in [a.strip() for a in args.attacks.split(",") if a.strip()]:
        t0 = time.time()
        rec = {t: {"true": [], "pred": [], "l2": [], "ssim": []} for t in models}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if atk in ("I-FGSM", "NAE"):
                src = {source: models[source]}
            elif atk == "DI-NI-FGSM":
                src = {source: models[source]}
            else:
                src = ens
            xa = A.run(atk, src, x, y, cfg)
            l2 = (xa - x).flatten(1).norm(dim=1).cpu().numpy()
            ss = ssim_mean(x, xa)
            for nm, m in models.items():
                pr = predict(m, xa).cpu().numpy()
                yy = y.cpu().numpy()
                rec[nm]["true"].append(yy)
                rec[nm]["pred"].append(pr)
                rec[nm]["l2"].append(l2)
                rec[nm]["ssim"].append(np.full(len(yy), ss))
        dt = time.time() - t0
        per = dt / len(test_items)
        for nm in models.items():
            pass
        for nm in models:
            np.savez_compressed(os.path.join(RES, f"{args.out}_{atk}_{nm}.npz"),
                                true=np.concatenate(rec[nm]["true"]),
                                pred=np.concatenate(rec[nm]["pred"]),
                                l2=np.concatenate(rec[nm]["l2"]))
        for nm in models:
            tr = np.concatenate(rec[nm]["true"])
            pr = np.concatenate(rec[nm]["pred"])
            met = S.metrics_from_preds(tr, pr, W)
            role = "集成内" if nm in ens_names else "未见"
            rows.append({
                "attack": atk, "target": nm, "role": role,
                "clean_acc": clean[nm], "ASR": met["ASR"],
                "SW-ASR": met["SW-ASR"], "DMR": met["DMR"],
                "L2": float(np.concatenate(rec[nm]["l2"]).mean()),
                "SSIM": float(np.concatenate(rec[nm]["ssim"]).mean()),
                "sec_per_sample": per,
            })
        print(f"  {atk:<12s} 完成 {dt:.1f}s（{per*1000:.1f} ms/样本）")

    csv_path = os.path.join(RES, f"{args.out}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # 按「集成内平均 / 未见平均」聚合
    print("\n== 汇总（各维度均值）==")
    print(f"{'攻击':<12s}{'集成内 ASR':>11s}{'SW-ASR':>9s}{'DMR':>8s}{'未见 ASR':>10s}"
          f"{'未见 SW':>9s}{'L2':>7s}{'SSIM':>7s}{'ms/样本':>9s}")
    summ = []
    for atk in [a.strip() for a in args.attacks.split(",") if a.strip()]:
        sub = [r for r in rows if r["attack"] == atk]
        ins = [r for r in sub if r["role"] == "集成内"]
        out = [r for r in sub if r["role"] == "未见"]
        f_ = lambda rs, k: sum(r[k] for r in rs) / len(rs) if rs else float("nan")
        line = {
            "attack": atk,
            "in_ASR": f_(ins, "ASR"), "in_SW": f_(ins, "SW-ASR"), "in_DMR": f_(ins, "DMR"),
            "out_ASR": f_(out, "ASR"), "out_SW": f_(out, "SW-ASR"),
            "L2": f_(sub, "L2"), "SSIM": f_(sub, "SSIM"),
            "ms": f_(sub, "sec_per_sample") * 1000,
        }
        summ.append(line)
        print(f"{atk:<12s}{line['in_ASR']:>11.4f}{line['in_SW']:>9.4f}{line['in_DMR']:>8.4f}"
              f"{line['out_ASR']:>10.4f}{line['out_SW']:>9.4f}{line['L2']:>7.2f}"
              f"{line['SSIM']:>7.3f}{line['ms']:>9.1f}")

    json_path = os.path.join(RES, f"{args.out}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "clean_acc": clean, "summary": summ, "rows": rows},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n明细 {csv_path}\n汇总 {json_path}")


if __name__ == "__main__":
    main()
