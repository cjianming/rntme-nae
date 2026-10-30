# -*- coding: utf-8 -*-
"""检测任务数据集适配：VOC（试点）与 KITTI（正式实验）。

两类数据的标注都映射到 **COCO 标签 id**，因为目标检测器用的是 torchvision 的
COCO 预训练权重，输出的就是 COCO id。映射表见 VOC2COCO / KITTI2COCO。

只保留交通场景真正关心的类，以减少 CPU 上的无效计算：
    person(1) / bicycle(2) / car(3) / motorcycle(4) / bus(6)

返回的样本：
    img    float32 (3,H,W) ∈ [0,1]   RGB
    boxes  float32 (N,4)  xyxy 绝对像素（已缩放到当前输入尺寸）
    labels int64   (N,)   COCO id
    ignore int64   (N,)   1 表示 difficult/忽略区，只用于不计 FP
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ------------------------------------------------------------ 类别映射
VOC_CLASSES = ["__bg__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
VOC2COCO = {
    "person": 1, "bicycle": 2, "car": 3, "motorbike": 4,
    "bus": 6, "train": 7, "truck": 8,
}
KITTI2COCO = {
    "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2,
    "Car": 3, "Van": 3, "Truck": 8, "Tram": 6, "Misc": -1, "DontCare": -1,
}
COCO_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 7: "train", 8: "truck"}


def _letterbox_box(b: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = b.copy().astype(np.float32)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


class DetDataset:
    """统一接口。kind ∈ {'voc', 'kitti'}。

    size=(H, W) 会把图直接 resize 到该尺寸（不做保持长宽比的 padding），
    框按比例同步缩放。检测器内部还会再 resize，但输入分辨率由这里决定，
    因而对抗扰动的分辨率也就是 (H, W)。
    """

    def __init__(self, kind: str = "voc", root: str = "data", size=(384, 512),
                 classes: Sequence[int] = (1, 3), split: str = "test",
                 max_images: int | None = None, seed: int = 0,
                 min_boxes: int = 1, cache: bool = True):
        self.kind = kind.lower()
        self.root = root
        self.size = size
        self.classes = list(classes)
        self.split = split
        self.seed = seed
        self.items: List[Tuple[str, List[dict]]] = []

        if self.kind == "voc":
            self._scan_voc(max_images)
        elif self.kind == "kitti":
            self._scan_kitti(max_images)
        else:
            raise ValueError(f"未知数据集 {kind}")

        if len(self.items) == 0:
            raise RuntimeError("没有可用的样本，检查数据路径与 classes 过滤条件")

        self.cache = None
        if cache:
            self.cache = np.stack([self._load_raw(p) for p, _ in self.items], 0)

    # -------------------------------------------------- 扫描
    def _scan_voc(self, max_images: int | None) -> None:
        from torchvision.datasets import VOCDetection
        year = "2007"
        ds = VOCDetection(root=self.root, year=year, image_set=self.split, download=True)
        idx = list(range(len(ds)))
        random.Random(self.seed).shuffle(idx)
        for i in idx:
            path, ann = ds[i]
            objs = ann["annotation"].get("object", [])
            if isinstance(objs, dict):
                objs = [objs]
            boxes, labels, ign = [], [], []
            for o in objs:
                nm = o["name"]
                if nm not in VOC2COCO:
                    continue
                c = VOC2COCO[nm]
                if c not in self.classes:
                    continue
                bb = o["bndbox"]
                boxes.append([float(bb["xmin"]), float(bb["ymin"]),
                              float(bb["xmax"]), float(bb["ymax"])])
                labels.append(c)
                ign.append(1 if str(o.get("difficult", "0")) == "1" else 0)
            if len(boxes) < 1:
                continue
            self.items.append((path, [
                {"boxes": np.array(boxes, np.float32),
                 "labels": np.array(labels, np.int64),
                 "ignore": np.array(ign, np.int64)}]))
            if max_images and len(self.items) >= max_images:
                break

    def _scan_kitti(self, max_images: int | None) -> None:
        """KITTI object detection 布局：
        root/kitti/training/image_2/*.png
        root/kitti/training/label_2/*.txt
        """
        base = os.path.join(self.root, "kitti", "training")
        img_dir = os.path.join(base, "image_2")
        lab_dir = os.path.join(base, "label_2")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"找不到 {img_dir}")
        names = sorted(f[:-4] for f in os.listdir(lab_dir) if f.endswith(".txt"))
        random.Random(self.seed).shuffle(names)
        for stem in names:
            boxes, labels, ign = [], [], []
            with open(os.path.join(lab_dir, stem + ".txt"), encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 8:
                        continue
                    c = KITTI2COCO.get(p[0], -1)
                    if c < 0 or c not in self.classes:
                        continue
                    if p[0] == "DontCare":
                        continue
                    boxes.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
                    labels.append(c)
                    ign.append(1 if float(p[1]) > 0.5 else 0)   # truncation>0.5 视为忽略
            if not boxes:
                continue
            self.items.append((os.path.join(img_dir, stem + ".png"), [
                {"boxes": np.array(boxes, np.float32),
                 "labels": np.array(labels, np.int64),
                 "ignore": np.array(ign, np.int64)}]))
            if max_images and len(self.items) >= max_images:
                break

    # -------------------------------------------------- 读取
    def _load_raw(self, path: str) -> np.ndarray:
        im = Image.open(path).convert("RGB")
        im = TF.resize(im, list(self.size), antialias=True)
        return np.asarray(im, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        path, meta = self.items[i]
        if self.cache is not None:
            arr = self.cache[i]
            im0 = Image.fromarray(arr)
            ow, oh = arr.shape[1], arr.shape[0]
        else:
            im0 = Image.open(path).convert("RGB")
            ow, oh = im0.size
            im0 = TF.resize(im0, list(self.size), antialias=True)
        m = meta[0]
        # 原图尺寸 -> 当前输入尺寸
        sx = self.size[1] / ow
        sy = self.size[0] / oh
        b = _letterbox_box(m["boxes"], sx, sy)
        img = torch.from_numpy(np.asarray(im0, np.uint8)).permute(2, 0, 1).float() / 255.0
        return {"img": img, "boxes": torch.from_numpy(b),
                "labels": torch.from_numpy(m["labels"]),
                "ignore": torch.from_numpy(m["ignore"])}

    def subset(self, idx: Sequence[int]) -> "Subset":
        return Subset(self, idx)


class Subset:
    def __init__(self, base: DetDataset, idx: Sequence[int]):
        self.base = base
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]

    @property
    def items(self):
        return [self.base.items[i] for i in self.idx]


def collate(samples: List[Dict[str, torch.Tensor]]):
    """检测器的 targets 是 List[Dict]，不能 stack。"""
    imgs = torch.stack([s["img"] for s in samples], 0)
    tg = [{"boxes": s["boxes"], "labels": s["labels"]} for s in samples]
    ign = [s["ignore"] for s in samples]
    return imgs, tg, ign


if __name__ == "__main__":
    ds = DetDataset(kind="voc", root="data", size=(384, 512), max_images=8, cache=False)
    print(f"样本 {len(ds)}")
    s = ds[0]
    print("img", tuple(s["img"].shape), s["img"].min().item(), s["img"].max().item())
    print("boxes", s["boxes"].shape, s["labels"].tolist())
    print("路径样例", ds.items[0][0])
