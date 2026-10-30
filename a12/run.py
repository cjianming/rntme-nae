# -*- coding: utf-8 -*-
"""A1.2 主实验：目标检测任务上的对抗迁移攻击。

流程：
    1. 取数据子集（synth = 合成交通标志场景 / voc / kitti = 真实数据）
    2. 干净预测，作为 mAP 与漏检率的基线
    3. 每种攻击：在「集成检测器」上生成扰动 -> 在集成内与未见检测器上评估
    4. 输出 CSV / JSON / npz（逐图检测结果，便于复核与重算）

用法（试点，CPU，合成场景 + 微调后的 SSDLite/FasterRCNN）：
    python -m a12.run --data synth --n 80 --T 10 --B 3 \\
        --ens ssdlite_a,ssdlite_b --heldout frcnn_mnv3 \\
        --ckpt ssdlite_a=a12/chk/ssdlite_a.pt,ssdlite_b=a12/chk/ssdlite_b.pt,\\
frcnn_mnv3=a12/chk/frcnn_mnv3.pt --tag pilot

用法（正式，GPU，KITTI 零样本）：
    python -m a12.run --data kitti --root data --n 1000 --size 384x1280 --T 20 --B 5 \\
        --ens ssdlite,frcnn_mnv3,retina_r50 --heldout frcnn_r50 --tag full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from . import attacks as A
from . import metrics as M
from .dataset import COCO_NAMES, DetDataset
from .detectors import Detector, build

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def parse_size(s: str):
    a, b = s.lower().split("x")
    return int(a), int(b)


def parse_ckpt(s: str) -> Dict[str, str]:
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def to_np(o: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {
        "boxes": o["boxes"].cpu().numpy().astype(np.float32),
        "scores": o["scores"].cpu().numpy().astype(np.float32),
        "labels": o["labels"].cpu().numpy().astype(np.int64),
    }


def predict_all(dets: Dict[str, Detector], imgs: List[torch.Tensor], bs: int) -> Dict[str, List[Dict]]:
    out = {n: [] for n in dets}
    for i in range(0, len(imgs), bs):
        xb = torch.cat(imgs[i:i + bs], 0)
        for n, d in dets.items():
            out[n].extend([to_np(o) for o in d.predict(xb)])
    return out


def class_name(c: int, synth: bool) -> str:
    if synth:
        from .synth import GROUP_NAMES
        return GROUP_NAMES[c - 1] if 1 <= c <= len(GROUP_NAMES) else str(c)
    return COCO_NAMES.get(c, str(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth", choices=["synth", "voc", "kitti"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--size", default="384x512")
    ap.add_argument("--classes", default="", help="留空则按数据集自动")
    ap.add_argument("--mr-class", type=int, default=1, help="单列漏检率的类 id")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--ens", default="ssdlite_a,ssdlite_b")
    ap.add_argument("--heldout", default="frcnn_mnv3")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--attacks", default="I-FGSM,NAE,DI-NI-FGSM,RNTME-NAE")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--alpha", type=float, default=1 / 255)
    ap.add_argument("--mu", type=float, default=0.9)
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--warp-boxes", type=int, default=1, help="消融：0=几何变换不同步 GT 框")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    hw = parse_size(args.size)
    is_synth = args.data == "synth"
    classes = [int(c) for c in args.classes.split(",")] if args.classes else \
        (list(range(1, 6)) if is_synth else [1, 3])

    # ---------------- 数据
    if is_synth:
        from .train import SynthSet
        ds = SynthSet("test", args.n, size=hw, seed=args.data_seed + 7777)
        samples = [ds[i] for i in range(len(ds))]
        imgs = [s["img"].unsqueeze(0) for s in samples]
        gts = [{"boxes": s["boxes"].numpy(), "labels": s["labels"].numpy(),
                "ignore": np.zeros(len(s["boxes"]), bool)} for s in samples]
        targets = [{"boxes": s["boxes"], "labels": s["labels"]} for s in samples]
    else:
        ds = DetDataset(kind=args.data, root=args.root, size=hw, classes=classes,
                        split=args.split, max_images=args.n, seed=args.data_seed)
        n0 = len(ds)
        imgs, gts, targets = [], [], []
        for i in range(n0):
            s = ds[i]
            imgs.append(s["img"].unsqueeze(0))
            gts.append({"boxes": s["boxes"].numpy(), "labels": s["labels"].numpy(),
                        "ignore": s["ignore"].numpy().astype(bool)})
            targets.append({"boxes": s["boxes"], "labels": s["labels"]})
    n = len(imgs)
    n_inst = sum(len(g["labels"]) for g in gts)
    print(f"[1/4] {args.data}: {n} 张 / {n_inst} 个目标 / 类 {classes}")

    # ---------------- 检测器
    ens_names = [s.strip() for s in args.ens.split(",") if s.strip()]
    held_names = [s.strip() for s in args.heldout.split(",") if s.strip()]
    print("[2/4] 构建检测器")
    dets = build(ens_names + held_names, ckpts=parse_ckpt(args.ckpt))

    print("[3/4] 干净基线")
    clean = predict_all(dets, imgs, args.bs)
    base = {nm: M.evaluate(clean[nm], gts, classes) for nm in dets}
    for nm in dets:
        r = base[nm]
        print(f"      {nm:<14s} mAP@0.5={r['mAP']:.4f}  " +
              "  ".join(f"R[{class_name(c, is_synth)}]={r['recall'][c]:.3f}" for c in classes))

    rows = []
    cfg_common = dict(eps=args.eps, alpha=args.alpha, T=args.T, mu=args.mu,
                      B=args.B, sigma=args.sigma, tau=2.0, beta=0.7,
                      gamma_geo=1.0, gamma_col=3.0, seed=args.seed, rel_norm=True,
                      warp_boxes=bool(args.warp_boxes))
    ens_dets = {k: dets[k] for k in ens_names}

    print("[4/4] 攻击与评估")
    for atk in [a.strip() for a in args.attacks.split(",") if a.strip()]:
        t0 = time.time()
        advs = []
        for i in range(n):
            xa = A.run(atk, ens_dets, imgs[i], [targets[i]], cfg_common)
            advs.append(xa.detach())
            if (i + 1) % 10 == 0:
                print(f"      {atk} 生成 {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
        gen = time.time() - t0
        adv = predict_all(dets, advs, args.bs)

        for nm in dets:
            r = M.evaluate(adv[nm], gts, classes)
            row = {
                "attack": atk, "detector": nm,
                "role": "集成内" if nm in ens_names else "未见",
                "mAP_clean": base[nm]["mAP"], "mAP_adv": r["mAP"],
                "d_mAP": base[nm]["mAP"] - r["mAP"],
                "drop_pct": 100 * (1 - r["mAP"] / base[nm]["mAP"]) if base[nm]["mAP"] > 0 else float("nan"),
                "ndet_clean": base[nm]["ndet"], "ndet_adv": r["ndet"],
                "sec_gen": gen / n,
            }
            c = args.mr_class
            row["MR_clean"] = 1 - base[nm]["recall"][c]
            row["MR_adv"] = 1 - r["recall"][c]
            row["dMR"] = row["MR_adv"] - row["MR_clean"]
            rows.append(row)
            print(f"      {atk:<11s} {nm:<14s} mAP {base[nm]['mAP']:.3f}->{r['mAP']:.3f} "
                  f"({row['drop_pct']:5.1f}%)  漏检 {row['MR_clean']:.3f}->{row['MR_adv']:.3f}  "
                  f"{gen/n:.2f}s/图", flush=True)

        # 逐图检测结果落盘（object 数组，压缩保存）
        payload = {}
        for nm in dets:
            for f in ("boxes", "scores", "labels"):
                payload[f"{atk}_{nm}_{f}"] = np.array(
                    [d[f] for d in adv[nm]], dtype=object)
        np.savez_compressed(os.path.join(RES, f"{args.tag}_{atk}.npz"), **payload)

    payload = {}
    for nm in dets:
        for f in ("boxes", "scores", "labels"):
            payload[f"clean_{nm}_{f}"] = np.array([d[f] for d in clean[nm]], dtype=object)
    np.savez_compressed(os.path.join(RES, f"{args.tag}_clean.npz"), **payload)

    csv_path = os.path.join(RES, f"{args.tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    json_path = os.path.join(RES, f"{args.tag}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "n_images": n, "n_instances": n_inst,
                   "classes": classes,
                   "clean": {nm: {"mAP": base[nm]["mAP"],
                                  "recall": {str(k): v for k, v in base[nm]["recall"].items()},
                                  "ndet": base[nm]["ndet"]} for nm in dets},
                   "rows": rows}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n明细 {csv_path}\n汇总 {json_path}")


if __name__ == "__main__":
    main()
