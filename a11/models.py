# -*- coding: utf-8 -*-
"""模型池。

两套：
  small —— 4 个架构互异的小网络，32x32 输入，CPU 可训，用于链路试点；
  full  —— 论文正式实验用的 5 个替代模型 + 2 个未见目标（torchvision 主干）。

论文里 M=5 的集成成员：resnet50 / efficientnet_b0 / vgg19 / mobilenet_v2 / googlenet
未见目标（完全不参与生成）：resnet101 / densenet121
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL = ["sresnet", "svgg", "ssep", "sinc"]
FULL_SURROGATE = ["resnet50", "efficientnet_b0", "vgg19", "mobilenet_v2", "googlenet"]
FULL_HELDOUT = ["resnet101", "densenet121"]


# ------------------------------------------------------------ 小网络
class _Basic(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.sc is None else self.sc(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + r)


class SmallResNet(nn.Module):
    """残差连接范式。"""
    def __init__(self, nc=43, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(True))
        self.s1 = _Basic(w, w)
        self.s2 = _Basic(w, w * 2, 2)
        self.s3 = _Basic(w * 2, w * 4, 2)
        self.s4 = _Basic(w * 4, w * 4, 2)
        self.fc = nn.Linear(w * 4, nc)

    def forward(self, x):
        x = self.s4(self.s3(self.s2(self.s1(self.stem(x)))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class SmallVGG(nn.Module):
    """深层序列卷积范式。"""
    def __init__(self, nc=43, w=32):
        super().__init__()
        def blk(i, o, n=2):
            layers = []
            for k in range(n):
                layers += [nn.Conv2d(i if k == 0 else o, o, 3, 1, 1, bias=False),
                           nn.BatchNorm2d(o), nn.ReLU(True)]
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        self.f = nn.Sequential(blk(3, w), blk(w, w * 2), blk(w * 2, w * 4), blk(w * 4, w * 4))
        self.fc = nn.Linear(w * 4, nc)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)
        return self.fc(x)


class _Sep(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class SmallSepNet(nn.Module):
    """深度可分离卷积范式（轻量）。"""
    def __init__(self, nc=43, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(True))
        self.f = nn.Sequential(_Sep(w, w * 2, 2), _Sep(w * 2, w * 4, 2),
                               _Sep(w * 4, w * 8, 2), _Sep(w * 8, w * 8, 2))
        self.fc = nn.Linear(w * 8, nc)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.f(self.stem(x)), 1).flatten(1)
        return self.fc(x)


class _IncCell(nn.Module):
    """Inception 单元：1x1 / 3x3 / 5x5 / pool 四支并行，拼接后归一化。"""
    def __init__(self, cin, cout, pool=True):
        super().__init__()
        b = cout // 4
        self.a = nn.Sequential(nn.Conv2d(cin, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))
        self.b = nn.Sequential(nn.Conv2d(cin, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))
        self.c = nn.Sequential(nn.Conv2d(cin, b, 5, 1, 2, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))
        self.d = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                               nn.Conv2d(cin, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))
        self.bn = nn.BatchNorm2d(b * 4)
        self.pool = pool

    def forward(self, x):
        y = torch.cat([self.a(x), self.b(x), self.c(x), self.d(x)], 1)
        y = F.relu(self.bn(y))
        return F.max_pool2d(y, 2) if self.pool else y


class SmallInception(nn.Module):
    """多分支并行范式。"""
    def __init__(self, nc=43, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(True))
        self.c1 = _IncCell(w, w * 2, True)
        self.c2 = _IncCell(w * 2, w * 4, True)
        self.c3 = _IncCell(w * 4, w * 8, True)
        self.fc = nn.Linear(w * 8, nc)

    def forward(self, x):
        x = self.c3(self.c2(self.c1(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ------------------------------------------------------------ torchvision
def _tv(name: str, nc: int, size: int):
    import torchvision.models as tvm
    m = getattr(tvm, name)(weights=None)
    if name.startswith("resnet") or name.startswith("densenet"):
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif name in ("vgg19", "vgg16"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    elif name in ("mobilenet_v2",):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    elif name in ("efficientnet_b0", "efficientnet_b1"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    elif name == "googlenet":
        m.fc = nn.Linear(m.fc.in_features, nc)
    else:
        raise ValueError(name)

    if size <= 64:
        # 小图输入：把首个 7x7/stride2 换成 3x3/stride1，去掉 maxpool 的过度下采样
        if hasattr(m, "conv1") and isinstance(m.conv1, nn.Conv2d):
            m.conv1 = nn.Conv2d(3, m.conv1.out_channels, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
        elif hasattr(m, "features") and isinstance(m.features[0], nn.Conv2d):
            m.features[0] = nn.Conv2d(3, m.features[0].out_channels, 3, 1, 1,
                                      bias=m.features[0].bias is not None)
        if name == "googlenet":
            m.transform_input = False
    return m


def build(name: str, nc: int = 43, size: int = 32) -> nn.Module:
    if name == "sresnet":
        return SmallResNet(nc)
    if name == "svgg":
        return SmallVGG(nc)
    if name == "ssep":
        return SmallSepNet(nc)
    if name == "sinc":
        return SmallInception(nc)
    return _tv(name, nc, size)


def describe(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
