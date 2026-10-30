# -*- coding: utf-8 -*-
"""GTSRB 读取与划分。裁剪 ROI 后缩放到 32（试点）或 64/224（全量）。"""
from __future__ import annotations

import csv
import os
import random
import zipfile
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image


def ensure_extracted(root: str) -> str:
    """把 data/gtsrb 下的三个 zip 解压到 data/gtsrb/extract，返回该目录。"""
    out = os.path.join(root, "extract")
    marker = os.path.join(out, ".done")
    if os.path.exists(marker):
        return out
    os.makedirs(out, exist_ok=True)
    zips = [f for f in os.listdir(root) if f.lower().endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"{root} 下没有 zip，请先下载 GTSRB 三个压缩包")
    for z in zips:
        with zipfile.ZipFile(os.path.join(root, z)) as zf:
            zf.extractall(out)
    open(marker, "w").close()
    return out


def _read_index(extracted: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """返回 (train_items, test_items)，每个 item 是 (图片路径, 真实类)。"""
    train_root = os.path.join(extracted, "GTSRB", "Final_Training", "Images")
    train: List[Tuple[str, int]] = []
    for cid in range(43):
        sub = os.path.join(train_root, f"{cid:05d}")
        if not os.path.isdir(sub):
            continue
        for fn in sorted(os.listdir(sub)):
            if fn.lower().endswith((".ppm", ".png", ".jpg")):
                train.append((os.path.join(sub, fn), cid))

    test_root = os.path.join(extracted, "GTSRB", "Final_Test", "Images")
    gt = None
    for cand in (os.path.join(extracted, "GT-final_test.csv"),
                 os.path.join(extracted, "GTSRB", "GT-final_test.csv"),
                 os.path.join(extracted, "GTSRB", "GTSRB_Final_Test_GT", "GT-final_test.csv"),
                 os.path.join(extracted, "GTSRB", "Final_Test", "GT-final_test.csv")):
        if os.path.exists(cand):
            gt = cand
            break
    if gt is None:                                   # 兜底：全目录搜一次
        for r, _, fs in os.walk(extracted):
            for fn in fs:
                if fn.lower() == "gt-final_test.csv":
                    gt = os.path.join(r, fn)
                    break
            if gt:
                break
    test: List[Tuple[str, int]] = []
    if gt:
        with open(gt, encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f, delimiter=";"))
        head = rows[0]
        ci = head.index("ClassId") if "ClassId" in head else len(head) - 1
        fi = head.index("Filename") if "Filename" in head else 0
        for r in rows[1:]:
            if len(r) <= max(ci, fi):
                continue
            test.append((os.path.join(test_root, r[fi]), int(r[ci])))
    else:
        raise FileNotFoundError("找不到 GT-final_test.csv")
    return train, test


class GTSRB:
    """轻量数据集：按需读图、裁 ROI、缩放。

    size: 32（CPU 试点）/ 64 / 224（全量 torchvision 主干）
    cache: 一次性把全部图读入内存（uint8）。PPM 逐张解码很慢，
           一个 epoch 重开上万张会成为瓶颈，训练时务必打开。
    """

    def __init__(self, items: List[Tuple[str, int]], size: int = 32,
                 train: bool = False, cache: bool = False):
        self.items = items
        self.size = size
        self.train = train
        self._cache = None
        if cache:
            import time as _t
            t0 = _t.time()
            arr = np.empty((len(items), size, size, 3), dtype=np.uint8)
            for i in range(len(items)):
                im = self._load(items[i][0])
                arr[i] = np.asarray(im, dtype=np.uint8)
            self._cache = arr
            print(f"    缓存 {len(items)} 张 {size}px 图，用时 {_t.time()-t0:.1f}s，"
                  f"占用 {arr.nbytes/1e6:.0f} MB")

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: str) -> Image.Image:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        # GTSRB 的标志大致居中，固定裁掉 10% 边框即可，避免依赖 GT 里逐图的 ROI
        m = 0.10
        im = im.crop((int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))))
        return im.resize((self.size, self.size), Image.BILINEAR)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        if self._cache is not None:
            x = self._cache[i].astype(np.float32) / 255.0
        else:
            x = np.asarray(self._load(path), dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5                      # -> [-1, 1]
        x = np.transpose(x, (2, 0, 1))
        return x, y


def split_train(items: List[Tuple[str, int]], val_frac: float = 0.1,
                seed: int = 0) -> Tuple[List, List]:
    """按类别分层切出验证集。"""
    rng = random.Random(seed)
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    tr, va = [], []
    for c, lst in by_cls.items():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, int(len(lst) * val_frac))
        va += lst[:k]
        tr += lst[k:]
    return tr, va


def stratified_subset(items: List[Tuple[str, int]], n: int, seed: int = 0) -> List:
    """按类别等比例抽样 n 张，保证 43 类都有。"""
    if n >= len(items):
        return items[:]
    rng = random.Random(seed)
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    out = []
    for c, lst in by_cls.items():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, round(len(lst) / len(items) * n))
        out += lst[:k]
    rng.shuffle(out)
    return out[:n]


def to_tensors(batch, device="cpu"):
    """把 __getitem__ 的 (ndarray, int) 列表拼成张量。调用方需已 import torch。"""
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    return xs, ys
