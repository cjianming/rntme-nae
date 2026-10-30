# -*- coding: utf-8 -*-
"""检测任务评价指标：mAP@0.5 与行人漏检率。

不依赖 pycocotools / pycocotools 的 VOC 版，全部自己实现，便于审稿人核对口径：
  * IoU 阈值为 0.5（单一阈值，与 VOC2007 主指标一致）
  * 按置信度降序做贪心匹配，一个 GT 只被匹配一次
  * AP 用 all-point 插值（continuous），非 11 点
  * difficult / 忽略类不参与评估，其上的检测也不计 FP（VOC 惯例）

指标定义（论文里要写清楚）：
  mAP@0.5     参与评估的类各自 AP 的算术平均
  MR_person   行人漏检率 = 1 - (IoU>=0.5 且 conf>=tau 的 GT 行人被检出比例)
  VDR         目标消失率：攻击后「原本被正确检出的目标」中不再被检出的比例
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------- IoU
def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) xyxy -> (N,M)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]                       # (N,M)
    aa = np.clip(a[:, 2:] - a[:, :2], 0, None)            # (N,2)
    bb = np.clip(b[:, 2:] - b[:, :2], 0, None)            # (M,2)
    area_a = (aa[:, 0] * aa[:, 1])[:, None]               # (N,1)
    area_b = (bb[:, 0] * bb[:, 1])[None, :]               # (1,M)
    ua = area_a + area_b - inter + 1e-9
    return (inter / ua).astype(np.float32)


def _as_np(t) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def match_dets(dets: np.ndarray, scores: np.ndarray, gts: np.ndarray,
               thr: float = 0.5) -> Tuple[np.ndarray, int]:
    """贪心匹配。返回 (每个检测的 tp 标记 0/1, 匹配到的 GT 数)。"""
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(gts), dtype=bool)
    tp = np.zeros(len(dets), dtype=np.int8)
    iou = box_iou_np(dets[order], gts)
    for k, i in enumerate(order):
        if len(gts) == 0:
            break
        row = iou[k].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= thr:
            used[j] = True
            tp[i] = 1
    return tp, int(used.sum())


# ---------------------------------------------------------------- AP
def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """all-point 插值 AP。tp 需按置信度降序排列。"""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp, dtype=np.float64)
    cfp = np.cumsum(1 - tp, dtype=np.float64)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # 单调化：从右向左取累积最大
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------- 汇总
def evaluate(dets_per_image: Sequence[Dict[str, np.ndarray]],
             gts_per_image: Sequence[Dict[str, np.ndarray]],
             classes: Sequence[int],
             conf: float = 0.05,
             iou_thr: float = 0.5) -> Dict[str, object]:
    """dets/gts 均为 [{'boxes':(N,4), 'scores':(N,), 'labels':(N,)}]。

    gts 额外允许 'ignore':(N,) bool，命中 ignore 的检测不计 FP。
    返回 {'AP': {cls: ap}, 'mAP': float, 'recall': {cls: r}, 'ndet': float}
    """
    AP: Dict[int, float] = {}
    REC: Dict[int, float] = {}
    n_det_total = 0

    for c in classes:
        # 收集该类的检测（按图拼接，记录图号以便匹配）
        tp_all: List[np.ndarray] = []
        sc_all: List[np.ndarray] = []
        n_gt = 0
        for d, g in zip(dets_per_image, gts_per_image):
            dl, gl = _as_np(d["labels"]), _as_np(g["labels"])
            db, ds = _as_np(d["boxes"]), _as_np(d["scores"])
            gb = _as_np(g["boxes"])
            keep = (dl == c) & (ds >= conf)
            db, ds = db[keep], ds[keep]
            gmask = gl == c
            if "ignore" in g:
                ign = _as_np(g["ignore"])
                gb_c, gb_ign = gb[gmask & ~ign], gb[gmask & ign]
            else:
                gb_c, gb_ign = gb[gmask], np.zeros((0, 4), np.float32)
            n_gt += len(gb_c)
            n_det_total += len(db)
            if len(db) == 0:
                continue
            order = np.argsort(-ds, kind="stable")
            tp, _ = match_dets(db, ds, gb_c, iou_thr)
            # 与 GT 重叠但落在 ignore 区、且未匹配上的检测，不计 FP
            if len(gb_ign) and len(db):
                iou_ign = box_iou_np(db, gb_ign).max(1)
                drop = (tp[order] == 0)
                drop_idx = order[drop]
                keep_fp = iou_ign[drop_idx] < iou_thr
                tp[drop_idx[~keep_fp]] = -1        # 标记为忽略
            valid = tp[order] >= 0
            tp_all.append(tp[order][valid])
            sc_all.append(ds[order][valid])

        if tp_all:
            tp_cat = np.concatenate(tp_all)
            sc_cat = np.concatenate(sc_all)
            perm = np.argsort(-sc_cat, kind="stable")
            AP[c] = average_precision(tp_cat[perm], n_gt)
        else:
            AP[c] = 0.0 if n_gt else float("nan")
        # recall：置信度 0.5 口径，衡量漏检
        n_hit = 0
        for d, g in zip(dets_per_image, gts_per_image):
            dl, gl = _as_np(d["labels"]), _as_np(g["labels"])
            db, ds = _as_np(d["boxes"]), _as_np(d["scores"])
            gb, gm = _as_np(g["boxes"]), gl == c
            gb_c = gb[gm]
            if not len(gb_c):
                continue
            m = (dl == c) & (ds >= 0.5)
            n_hit += match_dets(db[m], ds[m], gb_c, iou_thr)[1]
        REC[c] = n_hit / n_gt if n_gt else float("nan")

    vals = [v for v in AP.values() if not np.isnan(v)]
    return {
        "AP": AP,
        "mAP": float(np.mean(vals)) if vals else float("nan"),
        "recall": REC,
        "ndet": n_det_total,
    }


def attack_delta(clean: Dict[str, object], adv: Dict[str, object],
                 person_cls: int = 1) -> Dict[str, float]:
    """干净 vs 攻击后的差值，论文表 1 的行数据。"""
    return {
        "mAP_clean": clean["mAP"],
        "mAP_adv": adv["mAP"],
        "d_mAP": clean["mAP"] - adv["mAP"],
        "mAP_drop_pct": 100.0 * (1 - adv["mAP"] / clean["mAP"]) if clean["mAP"] > 0 else float("nan"),
        "MR_clean": 1 - clean["recall"][person_cls],
        "MR_adv": 1 - adv["recall"][person_cls],
        "d_MR": (1 - adv["recall"][person_cls]) - (1 - clean["recall"][person_cls]),
        "ndet_clean": clean["ndet"],
        "ndet_adv": adv["ndet"],
    }


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    # 构造一个小例子：2 个 GT，3 个检测（1 正确、1 错位、1 多余）
    gts = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 40, 40]], np.float32),
            "labels": np.array([1, 2]),
            "ignore": np.array([False, False])}]
    dets = [{"boxes": np.array([[0, 0, 10, 10], [21, 21, 39, 39], [100, 100, 120, 120]], np.float32),
             "scores": np.array([0.9, 0.8, 0.7], np.float32),
             "labels": np.array([1, 2, 1])}]
    r = evaluate(dets, gts, classes=[1, 2])
    print("AP", r["AP"], "mAP", round(r["mAP"], 4), "recall", r["recall"], "ndet", r["ndet"])
    assert abs(r["AP"][1] - 1.0) < 1e-6, "行人 AP 应为 1.0"
    assert abs(r["AP"][2] - 1.0) < 1e-6, "车辆 AP 应为 1.0"

    # 漏检：只留 1 个行人检测，AP 应下降
    dets2 = [{"boxes": np.array([[0, 0, 10, 10]], np.float32),
              "scores": np.array([0.9], np.float32),
              "labels": np.array([1])}]
    r2 = evaluate(dets2, gts, classes=[1, 2])
    print("漏检后 AP", r2["AP"], "mAP", round(r2["mAP"], 4))
    assert r2["AP"][2] == 0.0 and r2["AP"][1] == 1.0
    print("自检通过")
