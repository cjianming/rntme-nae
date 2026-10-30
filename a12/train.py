# -*- coding: utf-8 -*-
"""在合成的交通标志场景上微调检测器（COCO 预训练权重 -> 5 类检测头）。

正式实验不需要这个脚本：KITTI 上直接用 COCO 预训练检测器零样本评估即可
（KITTI 的行人/车辆本来就是 COCO 类）。这里微调完全是因为试点拿不到
真实带框图像，只能用合成场景。

改头方式：
    SSDLite   -> 替换 head.classification_head（SSDLiteClassificationHead）
    FasterRCNN-> 替换 roi_heads.box_predictor（FastRCNNPredictor）
类别数 = 5 + 1（背景）。合成标签 0..4 在喂给模型前统一 +1。

用法：
    python -m a12.train --arch ssdlite --n 400 --epochs 5 --out a12/chk/ssdlite_a.pt
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import metrics as M
from . import synth
from .detectors import BUILDERS, build_model as _bm

NUM_CLASSES = synth.NUM_GROUPS + 1          # +1 背景


# ------------------------------------------------------------ 数据
class SynthSet(Dataset):
    def __init__(self, split: str, n: int, size=(384, 512), seed: int = 0):
        self.items = synth.build(split, n, size=size, seed=seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.asarray(it["img"], dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)
        return {
            "img": img,
            "boxes": torch.from_numpy(it["boxes"]),
            "labels": torch.from_numpy(it["labels"] + 1),      # 0..4 -> 1..5
        }


def collate(batch):
    return ([b["img"] for b in batch],
            [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch])


# ------------------------------------------------------------ 改头
def build_model(arch: str) -> nn.Module:
    return _bm(arch, NUM_CLASSES)


# ------------------------------------------------------------ 评估
@torch.no_grad()
def evaluate(model, ds: SynthSet, classes: List[int], bs: int = 2) -> Dict:
    model.eval()
    dets, gts = [], []
    for i in range(0, len(ds), bs):
        batch = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        imgs = [b["img"] for b in batch]
        out = model(imgs)
        for o, b in zip(out, batch):
            dets.append({"boxes": o["boxes"].cpu().numpy(),
                         "scores": o["scores"].cpu().numpy(),
                         "labels": o["labels"].cpu().numpy()})
            gts.append({"boxes": b["boxes"].numpy(),
                        "labels": b["labels"].numpy(),
                        "ignore": np.zeros(len(b["boxes"]), bool)})
    return M.evaluate(dets, gts, classes)


# ------------------------------------------------------------ 训练
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ssdlite", choices=list(BUILDERS))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--nval", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--size", default="384x512")
    ap.add_argument("--seed", type=int, default=0, help="模型初始化/训练顺序种子")
    ap.add_argument("--data-seed", type=int, default=0, help="合成数据种子，同一批对比实验要相同")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="a12/chk/model.pt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    h, w = [int(v) for v in args.size.lower().split("x")]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[数据] 合成 {args.n} 训练 / {args.nval} 验证")
    tr = SynthSet("train", args.n, size=(h, w), seed=args.data_seed)
    va = SynthSet("test", args.nval, size=(h, w), seed=args.data_seed + 9999)
    print(f"       训练 {len(tr)} 张，验证 {len(va)} 张")

    print(f"[模型] {args.arch}，{NUM_CLASSES} 类")
    model = build_model(args.arch)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate,
                    num_workers=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    classes = list(range(1, NUM_CLASSES))
    best = -1.0

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for i, (imgs, tg) in enumerate(dl):
            loss = sum(v for v in model(imgs, tg).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            run += float(loss)
            if (i + 1) % 20 == 0:
                print(f"  ep{ep+1} {i+1}/{len(dl)} loss={run/(i+1):.3f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        sched.step()
        r = evaluate(model, va, classes)
        per = {synth.GROUP_NAMES[c - 1]: round(r["AP"][c], 3) for c in classes}
        print(f"[ep{ep+1}] loss={run/max(len(dl),1):.3f} mAP@0.5={r['mAP']:.4f} "
              f"({time.time()-t0:.0f}s) {per}", flush=True)
        if r["mAP"] > best:
            best = r["mAP"]
            torch.save({"arch": args.arch, "state": model.state_dict(),
                        "classes": NUM_CLASSES, "mAP": best}, args.out)
            print(f"       已保存 {args.out}（mAP {best:.4f}）", flush=True)

    print(f"\n完成。最佳 val mAP@0.5 = {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
