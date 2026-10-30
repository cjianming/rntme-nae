# -*- coding: utf-8 -*-
"""合成交通标志检测场景。

为什么要有这个：A1.2 正式实验的数据是 KITTI（dataset.py 已支持），但本机网络
到 KITTI / VOC / COCO 的镜像都不通，只能从 GTSRB 现成的 43 类标志贴图合成
带框场景。合成数据的绝对精度不能写进论文，它的用途有三个：

  1. 验证「检测版攻击」整条链路正确（几何变换必须同步变换 GT 框，否则 loss 全错）；
  2. 做「框同步 / 不框同步」的消融，这是实现正确性问题，与数据真实度无关；
  3. 观测「分类攻击直接搬到检测任务上会衰减多少」，这一结论与数据来源无关。

场景构图：
    背景：程序生成的天空—路面渐变 + 车道线 + 噪声（不用 GTSRB 原图当背景，
          避免标志纹理重复导致标签泄漏）
    前景：从 GTSRB 训练集随机取标志 ROI，随机缩放/旋转/调光后贴到画面
    标注：贴图外接框，5 个粗类

5 个粗类（把 43 细类按交通语义合并，便于训练小检测器）：
    0 限速    1 禁令    2 警告    3 指示    4 优先通行
"""
from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from a11 import data as G   # 复用 GTSRB 索引与解压逻辑

GROUP_NAMES = ["限速", "禁令", "警告", "指示", "优先通行"]
NUM_GROUPS = len(GROUP_NAMES)


def group_of(cid: int) -> int:
    """GTSRB 43 细类 -> 5 粗类。"""
    if cid in range(0, 9):          # 20~120 限速
        return 0
    if cid in (9, 10):              # 禁超及其解除
        return 1
    if cid in (11, 12):             # 优先权
        return 4
    if cid in (13, 14, 15, 16, 17):
        return 1                    # 让行/停车/禁行
    if 18 <= cid <= 31:
        return 2                    # 警告
    if 32 in (cid,) or cid in (41, 42):
        return 1                    # 解除类归入禁令
    return 3                        # 33~40 指示


def roi_map(gtsrb_extracted: str) -> Dict[str, Tuple[int, int, int, int]]:
    """读 GTSRB 训练集每类的 GT-xxxxx.csv，得到 {图片路径: ROI}。

    训练集的 ROI 标注按类分文件存放（GT-00000.csv ... GT-00042.csv），
    字段为 Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId。
    用精确 ROI 而不是固定裁边，贴图里才不会带半圈背景。
    """
    base = os.path.join(gtsrb_extracted, "GTSRB", "Final_Training", "Images")
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for cid in range(43):
        d = os.path.join(base, f"{cid:05d}")
        csv_path = os.path.join(d, f"GT-{cid:05d}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f, delimiter=";"))
        for r in rows[1:]:
            if len(r) < 8:
                continue
            out[os.path.join(d, r[0])] = (int(r[3]), int(r[4]), int(r[5]), int(r[6]))
    return out


_ROI_CACHE: Dict[str, Dict] = {}


def _read_sign(path: str, gtsrb_root: str = "data/gtsrb") -> np.ndarray:
    """按精确 ROI 裁出标志本体，返回 HxWx3 uint8。"""
    if gtsrb_root not in _ROI_CACHE:
        ext = G.ensure_extracted(gtsrb_root)
        _ROI_CACHE[gtsrb_root] = roi_map(ext)
    roi = _ROI_CACHE[gtsrb_root].get(path)
    im = Image.open(path).convert("RGB")
    if roi is None:
        w, h = im.size
        m = 0.10
        im = im.crop((int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))))
    else:
        im = im.crop(roi)
    return np.asarray(im, dtype=np.uint8)


# ------------------------------------------------------------ 背景
def _background(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    """天空—路面渐变 + 车道线 + 噪声。horizon 位置随机。"""
    horizon = int(h * rng.uniform(0.35, 0.6))
    img = np.zeros((h, w, 3), np.float32)

    sky_top = rng.uniform(110, 190)
    road = rng.uniform(55, 105)
    img[:horizon] = np.linspace(sky_top, sky_top * 0.8, horizon)[:, None, None] \
        + rng.normal(0, 4, (horizon, w, 3))
    img[horizon:] = np.linspace(road * 1.15, road * 0.7, h - horizon)[:, None, None] \
        + rng.normal(0, 6, (h - horizon, w, 3))

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    # 车道线：从消失点向外发散
    vp_x = int(w * rng.uniform(0.35, 0.65))
    for k in (-1, 0, 1):
        for seg in range(int(h - horizon) // 26):
            y0 = horizon + seg * 26 + rng.integers(0, 6)
            y1 = y0 + 14
            if y1 > h:
                break
            t0 = (y0 - horizon) / max(h - horizon, 1)
            t1 = (y1 - horizon) / max(h - horizon, 1)
            x0 = vp_x + k * (w * 0.52) * t0
            x1 = vp_x + k * (w * 0.52) * t1
            d.line([(x0, y0), (x1, y1)], fill=(225, 225, 210), width=max(2, int(1 + 4 * t0)))
    if rng.random() < 0.5:
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    return pil


# ------------------------------------------------------------ 标志贴图
def _paste_sign(canvas: Image.Image, sign: Image.Image, cx: int, cy: int,
                scale_h: int, angle: float, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """把标志按目标高度缩放、旋转后贴到 (cx, cy)，返回 bbox。"""
    sw, sh = sign.size
    nh = max(24, scale_h)
    nw = max(24, int(round(sw * nh / sh)))
    s = sign.resize((nw, nh), Image.BICUBIC)

    # 亮度/对比度扰动
    a = rng.uniform(0.82, 1.18)
    b = rng.uniform(-18, 18)
    arr = np.asarray(s, np.float32) * a + b
    s = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if abs(angle) > 0.5:
        mask = Image.new("L", s.size, 255)
        s = s.convert("RGBA").rotate(angle, resample=Image.BICUBIC, expand=True)
        mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)
        bw, bh = s.size
        x0 = max(0, min(int(cx - bw / 2), canvas.width - bw))
        y0 = max(0, min(int(cy - bh / 2), canvas.height - bh))
        canvas.paste(s.convert("RGB"), (x0, y0), mask)
        return x0, y0, x0 + bw, y0 + bh
    bw, bh = s.size
    x0 = max(0, min(int(cx - bw / 2), canvas.width - bw))
    y0 = max(0, min(int(cy - bh / 2), canvas.height - bh))
    canvas.paste(s, (x0, y0))
    return x0, y0, x0 + bw, y0 + bh


# ------------------------------------------------------------ 生成器
def build(split: str, n: int, size=(384, 512), seed: int = 0,
          signs_per_image: Tuple[int, int] = (1, 3),
          gtsrb_root: str = "data/gtsrb") -> List[Dict]:
    """返回 [{img: PIL.Image, boxes: (N,4), labels: (N,)}]。"""
    root = G.ensure_extracted(gtsrb_root)
    train_items, test_items = G._read_index(root)
    pool = test_items if split == "test" else train_items
    rng = np.random.default_rng(seed)

    # 预取一批标志 ROI（按类分组），避免每张都读盘
    by_group: Dict[int, List[np.ndarray]] = {g: [] for g in range(NUM_GROUPS)}
    pick = list(pool)
    random.Random(seed).shuffle(pick)
    for path, cid in pick[:4000]:
        by_group[group_of(cid)].append(_read_sign(path, gtsrb_root))
    by_group = {k: v for k, v in by_group.items() if v}
    if not by_group:
        raise RuntimeError("GTSRB 里没有可用标志")

    H, W = size
    out = []
    groups = list(by_group)
    for i in range(n):
        bg = _background(W, H, rng)
        k = int(rng.integers(signs_per_image[0], signs_per_image[1] + 1))
        boxes, labels = [], []
        used: List[Tuple[int, int, int, int]] = []
        for _ in range(k * 4):
            if len(boxes) >= k:
                break
            g = groups[int(rng.integers(0, len(groups)))]
            arr = by_group[g][int(rng.integers(0, len(by_group[g])))]
            sign = Image.fromarray(arr).convert("RGB")
            # 越远越小、越靠上
            sh = int(rng.uniform(46, 128))
            cy = int(rng.uniform(H * 0.18, H * 0.78))
            cx = int(rng.uniform(sh, W - sh))
            x0, y0, x1, y1 = _paste_sign(bg, sign, cx, cy, sh,
                                        float(rng.uniform(-12, 12)), rng)
            # 与已有框重叠过大则丢弃，重采
            ok = True
            for (a0, b0, a1, b1) in used:
                ov = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
                if ov > 0.25 * (x1 - x0) * (y1 - y0):
                    ok = False
                    break
            if not ok:
                continue
            used.append((x0, y0, x1, y1))
            boxes.append([x0, y0, x1, y1])
            labels.append(g)
        if not boxes:
            continue
        out.append({"img": bg,
                    "boxes": np.array(boxes, np.float32),
                    "labels": np.array(labels, np.int64)})
    return out


def dump(items: Sequence[Dict], dirpath: str, prefix: str = "") -> str:
    """落盘：图片 PNG + 一个 labels.csv（文件名,x1,y1,x2,y2,组）。"""
    os.makedirs(dirpath, exist_ok=True)
    rows = []
    for i, it in enumerate(items):
        fn = f"{prefix}{i:05d}.png"
        it["img"].save(os.path.join(dirpath, fn))
        for (b, l) in zip(it["boxes"], it["labels"]):
            rows.append([fn, *[f"{v:.1f}" for v in b], int(l)])
    csv_path = os.path.join(dirpath, f"labels{('_' + prefix) if prefix else ''}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "x1", "y1", "x2", "y2", "label"])
        w.writerows(rows)
    return csv_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    items = build("train", n, seed=1)
    print(f"生成 {len(items)} 张")
    c = {}
    for it in items:
        for l in it["labels"]:
            c[int(l)] = c.get(int(l), 0) + 1
    print("类别分布", {GROUP_NAMES[k]: v for k, v in sorted(c.items())})
    print("平均目标数", sum(len(i["labels"]) for i in items) / len(items))
