# -*- coding: utf-8 -*-
"""A3 模型级防御：PGD 对抗训练。

为什么要单独做这件事：预处理防御（JPEG/位深/模糊）在原理上只能削掉
扰动的高频成分，攻击者知道它们存在就能补偿；对抗训练是唯一被证明
能改变模型决策边界本身的做法。但它的代价也最直接——干净准确率。

本脚本训练一个与 A1.1 同架构、同数据、同轮数的 sresnet，唯一的区别是
训练时用 PGD 样本代替干净样本，然后回答三个问题：

  Q1 干净准确率掉多少？（防御的代价）
  Q2 原有的集成迁移攻击打在它身上还灵吗？（直接前向已保存的对抗样本）
  Q3 针对它重新做白盒攻击，能绕过去吗？（自适应攻击，重新生成扰动）

Q1/Q2 用现成的 npz 就能算，Q3 必须重新生成，因此比朴素评估慢一个数量级。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A          # noqa: E402
from a11 import data as DA           # noqa: E402
from a11 import models as M          # noqa: E402
from a11 import swasr as S           # noqa: E402
from a13 import defenses as DF       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHK = os.path.join(HERE, "chk")
RES = os.path.join(HERE, "results")


def pgd(model, x, y, eps, alpha, steps, rand_init=True, rng=None):
    """标准 PGD（L∞）。与 A1.1 的 I-FGSM 区别：随机初始化 + 多步。"""
    if rand_init:
        xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(-1, 1)
    else:
        xa = x.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        if xa.grad is not None:
            xa.grad.zero_()
        loss = F.cross_entropy(model(xa), y)
        g = torch.autograd.grad(loss, xa)[0]
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(-1, 1)
    return xa.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="sresnet")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02, help="对抗训练梯度噪声大，lr 要比干净训练小")
    p.add_argument("--train-subset", type=int, default=12000)
    p.add_argument("--test-subset", type=int, default=600)
    p.add_argument("--eps255", type=int, default=8, help="训练扰动上限，按 x/255 计")
    p.add_argument("--eps", type=float, default=None, help="直接给归一化值，覆盖 --eps255")
    p.add_argument("--steps", type=int, default=3, help="PGD 步数")
    p.add_argument("--freeze-bn", action="store_true",
                   help="冻结 BN 统计到干净数据（对抗训练轮数少时必需）")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", default="adv_sresnet")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(CHK, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    eps = args.eps if args.eps is not None else args.eps255 / 255.0
    alpha = eps * 2.5 / args.steps
    print(f"对抗训练 {args.arch} {args.size}px | eps={eps*255:.0f}/255 "
          f"alpha={alpha*255:.2f}/255 steps={args.steps} | {args.epochs} epoch")

    ext = DA.ensure_extracted(os.path.join(ROOT, "data", "gtsrb"))
    tr_items, te_items = DA._read_index(ext)
    if args.train_subset:
        tr_items = DA.stratified_subset(tr_items, args.train_subset, seed=1)
    tr = DataLoader(DA.GTSRB(tr_items, args.size, True, cache=True),
                    batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    te_items = DA.stratified_subset(te_items, args.test_subset, seed=2)
    te = DataLoader(DA.GTSRB(te_items, args.size, False, cache=True),
                    batch_size=256, shuffle=False, num_workers=0)

    model = M.build(args.arch, 43, args.size)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.freeze_bn:
        # 先用干净数据把 BN 的 running stats 校准到干净分布，之后冻结。
        # 不这么做，对抗样本会把 running_mean/var 拉偏，eval 时干净准确率会崩
        # （试点实测：不冻结时 6 轮后 eval 干净准确率只有 17%）。
        model.train()
        with torch.no_grad():
            for xb, _ in tr:
                model(xb)
        nb = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        print(f"  已用干净数据校准并冻结 {nb} 个 BN 层")

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        tot = cor = 0
        for x, y in tr:
            xa = pgd(model, x, y, eps, alpha, args.steps)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                cor += (model(xa).argmax(1) == y).sum().item()
                tot += y.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            vc = vn = 0
            for x, y in te:
                vc += (model(x).argmax(1) == y).sum().item()
                vn += y.numel()
        print(f"  ep {ep+1:02d}/{args.epochs}  adv_acc {cor/tot:.3f}  "
              f"clean {vc/vn:.3f}  {time.time()-t0:.0f}s", flush=True)

    torch.save({"arch": args.arch, "size": args.size, "state": model.state_dict(),
                "test_acc": vc / vn, "eps_train": float(eps), "steps": args.steps,
                "epochs": args.epochs}, os.path.join(CHK, f"{args.out}.pt"))
    print(f"  -> {os.path.join(CHK, args.out + '.pt')}")

    # ---------------- Q1 / Q2：干净与迁移
    model.eval()
    W = S.build_matrix()
    xs = torch.cat([x for x, _ in te])
    ys = torch.cat([y for _, y in te])
    rep = {"arch": args.arch, "eps_train": float(eps), "clean_acc": vc / vn,
           "transfer": {}, "adaptive": {}}
    print("\nQ1 干净准确率 "
          f"{vc/vn:.4f}（对照 sresnet 干净训练 0.958，见 A11 记录）")

    # 干净样本上的预测，用于算条件 ASR（只在本来分类对的样本上看攻击效果）
    with torch.no_grad():
        clean_pr = model(xs).argmax(1).numpy()
    ok = (clean_pr == ys.numpy())
    print(f"  干净正确样本 {ok.sum()}/{len(ok)}")

    def cond_metrics(y_true, pr, mask):
        """只保留 mask 内样本的指标；ASR 即条件攻击成功率。"""
        m = S.metrics_from_preds(np.asarray(y_true)[mask], np.asarray(pr)[mask], W)
        m["ASR_all"] = float((np.asarray(pr) != np.asarray(y_true)).mean())
        return m

    print("\nQ2 已有集成迁移攻击直接前向（括号内为条件 ASR）")
    for atk in ("I-FGSM", "NAE", "DI-NI-FGSM", "RNTME-NAE"):
        f = os.path.join(RES, f"a3_plain_{atk}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        xa = DF.from_uint8(z["x_adv"])
        with torch.no_grad():
            pr = model(xa).argmax(1).numpy()
        met = S.metrics_from_preds(z["y"], pr, W)
        cm = cond_metrics(z["y"], pr, ok)
        met["ASR_cond"] = cm["ASR"]
        rep["transfer"][atk] = met
        print(f"  {atk:<13s} ASR {met['ASR']:.3f} (条件 {cm['ASR']:.3f})  "
              f"SW-ASR {cm['SW-ASR']:.3f}  DMR {cm['DMR']:.3f}")

    # ---------------- Q3：自适应攻击（针对本模型重生成）
    print("\nQ3 针对本模型重做白盒攻击（自适应）")
    cfg = dict(eps=16 / 255, alpha=2 / 255, T=15, mu=0.9, source=args.arch,
               B=5, tau=2.0, beta=0.7, sigma=1.5, early_stop=False,
               seed=0, win=5, rho=0.9, delta=0.05)
    src = {args.arch: model}
    for atk in ("I-FGSM", "RNTME-NAE"):
        adv, yy = [], []
        t1 = time.time()
        for x, y in te:
            adv.append(DF.to_uint8(A.run(atk, src, x, y, cfg)))
            yy.append(y.numpy().astype(np.int64))
        xa = DF.from_uint8(np.concatenate(adv))
        with torch.no_grad():
            pr = model(xa).argmax(1).numpy()
        yy_c = np.concatenate(yy)
        met = S.metrics_from_preds(yy_c, pr, W)
        cm = cond_metrics(yy_c, pr, ok)
        met["ASR_cond"] = cm["ASR"]
        rep["adaptive"][atk] = met
        print(f"  {atk:<13s} ASR {met['ASR']:.3f} (条件 {cm['ASR']:.3f})  "
              f"SW-ASR {cm['SW-ASR']:.3f}  （{time.time()-t1:.0f}s）", flush=True)
        np.savez_compressed(os.path.join(RES, f"advtrain_adaptive_{atk}.npz"),
                            x_adv=np.concatenate(adv), y=np.concatenate(yy))

    with open(os.path.join(RES, "advtrain_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n写出 {os.path.join(RES, 'advtrain_summary.json')}")


if __name__ == "__main__":
    main()
