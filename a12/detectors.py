# -*- coding: utf-8 -*-
"""检测器封装：统一「求攻击损失」与「出检测框」两条路径。

为什么要单独封装：
  1. torchvision 的检测模型只有 train() 模式下才返回 loss dict，而 train() 会
     更新 BatchNorm 统计量、且 RPN 采样带随机性。攻击时既要拿到 loss，又不能
     让 BN 被对抗样本污染，所以统一 freeze BN，并按步重设随机种子。
  2. 各模型的输入尺寸策略不同（SSDLite 固定 320，Faster R-CNN 走 min_size），
     评估与攻击必须用同一套预处理，否则 mAP 不可比。

被试模型（试点配置，括号内为正式实验建议）：
  替代/集成成员：ssdlite320_mobilenet_v3_large、
                 fasterrcnn_mobilenet_v3_large_320_fpn
  未见目标：    fasterrcnn_resnet50_fpn_v2
  可选加：      retinanet_resnet50_fpn_v2（更强，CPU 慢，正式实验上）
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)

# 各模型在 CPU 上的相对开销（实测后用 probe.py 覆盖）
BUILDERS = {
    "ssdlite": lambda: ssdlite320_mobilenet_v3_large(weights="DEFAULT"),
    "frcnn_mnv3": lambda: fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT"),
    "frcnn_r50": lambda: fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT", min_size=400, max_size=512, box_score_thresh=0.05),
    "retina_r50": lambda: retinanet_resnet50_fpn_v2(
        weights="DEFAULT", min_size=400, max_size=512, score_thresh=0.05),
}


def _freeze_bn(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            mod.eval()
            mod.track_running_stats = True


def rehead(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """把 COCO 91 类检测头换成 num_classes（含背景）。"""
    if arch == "ssdlite":
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        old = model.head.classification_head
        in_ch = []
        for m in old.module_list:
            conv = next(l for l in m if isinstance(l, nn.Conv2d))
            in_ch.append(conv.in_channels)
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # norm_layer 会被 ConvNormActivation 以 norm_layer(channels) 调用，
        # 不能传 partial(nn.ReLU6, inplace=True)，必须包一层忽略通道数的 lambda。
        model.head.classification_head = SSDLiteClassificationHead(
            in_ch, num_anchors, num_classes, lambda _c: nn.ReLU6(inplace=True))
    else:
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def build_model(arch: str, num_classes: int | None = None, ckpt: str | None = None) -> nn.Module:
    """构建检测器；给定 ckpt 时按 ckpt 里记录的 arch/类别数重建并载入权重。"""
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu")
        arch = blob.get("arch", arch)
        num_classes = blob.get("classes", num_classes)
        m = rehead(BUILDERS[arch](), arch, num_classes)
        m.load_state_dict(blob["state"])
        return m
    m = BUILDERS[arch]()
    if num_classes is not None:
        m = rehead(m, arch, num_classes)
    return m


class Detector:
    """一个检测器的薄封装。name 可以是 BUILDERS 的键，也可以是自定义别名（配合 ckpt）。"""

    def __init__(self, name: str, device: str = "cpu", ckpt: str | None = None):
        self.name = name
        self.device = device
        self.ckpt = ckpt
        if ckpt:
            self.model = build_model(name, ckpt=ckpt).to(device)
        else:
            if name not in BUILDERS:
                raise KeyError(f"未知检测器 {name}，可选 {list(BUILDERS)}")
            self.model = BUILDERS[name]().to(device)
        self.model.eval()
        _freeze_bn(self.model)

    # ---------------- 推理路径
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        self.model.eval()
        out = self.model([xi for xi in x.to(self.device)])
        return [{k: v.detach().cpu() for k, v in o.items()} for o in out]

    # ---------------- 攻击路径
    def loss_sum(self, x: torch.Tensor, targets: Sequence[Dict[str, torch.Tensor]],
                 seed: int | None = None) -> torch.Tensor:
        """返回该检测器在 (x, targets) 上的总损失（标量），可反传到 x。

        检测损失 = 分类损失 + 回归损失（各模型自己的加权方式），不做归一化：
        我们要的是「让检测器整体失配」的方向，而非某个特定损失的最小化。
        """
        self.model.train()
        _freeze_bn(self.model)
        if seed is not None:
            torch.manual_seed(seed)
        tg = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        out = self.model(x.to(self.device), tg)
        if isinstance(out, dict):
            return sum(v for v in out.values() if torch.is_tensor(v))
        # 少数版本在 train 且未给 targets 时返回 list
        return torch.zeros((), device=self.device)

    def grads(self, x: torch.Tensor, targets, seed: int | None = None) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        loss = self.loss_sum(x, targets, seed)
        g, = torch.autograd.grad(loss, x)
        return g.detach()

    # ---------------- 杂项
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def build(names: Sequence[str], device: str = "cpu",
          ckpts: Dict[str, str] | None = None) -> Dict[str, Detector]:
    ckpts = ckpts or {}
    out = {}
    for n in names:
        t0 = time.time()
        out[n] = Detector(n, device, ckpts.get(n))
        tag = f"({os.path.basename(ckpts[n])})" if n in ckpts else ""
        print(f"  [det] {n:<14s}{out[n].n_params()/1e6:6.1f}M  "
              f"{time.time()-t0:.1f}s {tag}", flush=True)
    return out


def probe_speed(dets: Dict[str, Detector], hw=(384, 512), reps: int = 3) -> None:
    """测单图前向与单步梯度的时间，用来决定试点规模。"""
    x = torch.rand(1, 3, *hw)
    tg = [{"boxes": torch.tensor([[40.0, 40.0, 200.0, 220.0]]),
           "labels": torch.tensor([1])}]
    print(f"\n{'检测器':<14s}{'前向 ms':>10s}{'梯度 ms':>10s}")
    for n, d in dets.items():
        t0 = time.time()
        for _ in range(reps):
            d.predict(x)
        fw = (time.time() - t0) / reps * 1000
        t0 = time.time()
        for _ in range(reps):
            d.grads(x, tg)
        bw = (time.time() - t0) / reps * 1000
        print(f"{n:<14s}{fw:>10.0f}{bw:>10.0f}")


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["ssdlite", "frcnn_mnv3", "frcnn_r50"]
    print("构建检测器（首次会下载 COCO 预训练权重）")
    D = build(names)
    probe_speed(D)
