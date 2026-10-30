# -*- coding: utf-8 -*-
"""训练 GTSRB 分类器（替代模型 / 目标模型）。

用法（全量，GPU）：
    python -m a11.train --arch resnet50 --size 224 --epochs 30
    python -m a11.train --arch all --size 224 --epochs 30

用法（CPU 试点）：
    python -m a11.train --arch all --size 32 --epochs 10 --train-subset 12000 --cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import data as D          # noqa: E402
from a11 import models as M        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "chk")


def get_loaders(args):
    ext = D.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    train_items, test_items = D._read_index(ext)
    if args.train_subset and args.train_subset < len(train_items):
        train_items = D.stratified_subset(train_items, args.train_subset, seed=1)
    tr, va = D.split_train(train_items, 0.1, seed=0)
    if args.test_subset and args.test_subset < len(test_items):
        test_items = D.stratified_subset(test_items, args.test_subset, seed=2)
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(test_items)}，输入 {args.size}px")
    mk = lambda it, tf: DataLoader(D.GTSRB(it, args.size, tf, cache=True), batch_size=args.bs,
                                   shuffle=tf, num_workers=args.workers, drop_last=False)
    return mk(tr, True), mk(va, False), mk(test_items, False), len(test_items)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    c = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return c / max(n, 1)


def train_one(arch, args, loaders, device):
    tr, va, te, n_te = loaders
    model = M.build(arch, 43, args.size).to(device)
    nparam = M.describe(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = max(1, len(tr)) * args.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                              pct_start=0.15)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0, tot, correct = time.time(), 0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tot += y.numel()
            correct += (out.argmax(1) == y).sum().item()
        acc_va = accuracy(model, va, device)
        best = max(best, acc_va)
        print(f"  [{arch}] ep {ep+1:02d}/{args.epochs}  train {correct/tot:.3f}  "
              f"val {acc_va:.3f}  {time.time()-t0:.1f}s", flush=True)
        # 每轮落盘一次：训练若被中断，仍可用最后一轮权重（test_acc 记为 nan，训练结束后覆盖）
        os.makedirs(CKPT, exist_ok=True)
        torch.save({"arch": arch, "size": args.size, "state": model.state_dict(),
                    "test_acc": float("nan"), "val_best": best},
                   os.path.join(CKPT, f"{arch}_s{args.size}.pt"))
    acc_te = accuracy(model, te, device)
    print(f"  [{arch}] 完成：val_best {best:.4f}  test {acc_te:.4f}  参数 {nparam/1e6:.2f}M")
    os.makedirs(CKPT, exist_ok=True)
    path = os.path.join(CKPT, f"{arch}_s{args.size}.pt")
    torch.save({"arch": arch, "size": args.size, "state": model.state_dict(),
                "test_acc": acc_te, "val_best": best}, path)
    print(f"  -> {path}")
    return acc_te, nparam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="all",
                   help="all | 逗号分隔 | full（5个替代+2个未见）")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--train-subset", type=int, default=0)
    p.add_argument("--test-subset", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"设备 {device}，torch {torch.__version__}")
    loaders = get_loaders(args)

    if args.arch == "all":
        archs = M.SMALL if args.size <= 64 else M.FULL_SURROGATE + M.FULL_HELDOUT
    elif args.arch == "full":
        archs = M.FULL_SURROGATE + M.FULL_HELDOUT
    else:
        archs = [a.strip() for a in args.arch.split(",") if a.strip()]

    rows = []
    for a in archs:
        acc, npar = train_one(a, args, loaders, device)
        rows.append((a, acc, npar))
    print("\n== 汇总 ==")
    for a, acc, n in rows:
        print(f"  {a:<16s} test {acc:.4f}  {n/1e6:6.2f} M")


if __name__ == "__main__":
    main()
