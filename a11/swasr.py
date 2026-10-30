# -*- coding: utf-8 -*-
"""
SW-ASR / DMR：面向交通安全的攻击评价指标
=========================================

论文里 ASR 只回答「有没有骗过」，不回答「骗得有多危险」。
本模块把误分类按交通语义赋权，给出两个指标：

    SW-ASR  Severity-Weighted Attack Success Rate
            = (1/N) * Σ_i  w(t_i -> pred_i) * 1[pred_i != t_i]
            加权后的成功率，w∈[0,1] 表示「这次误判有多危险」

    DMR     Dangerous Misclassification Rate
            = (1/N) * Σ_i 1[ w(t_i -> pred_i) >= 0.8 ]
            把样本误判成「明显更宽松」类别的比例（只数严重的那一档）

权重构造原则：**只惩罚「朝更宽松方向」的错**。
把限速 80 读成 30 是效率损失，把限速 30 读成 80 是违法与事故，
两者不该同权。故 w 主要由「预测类相对真实类的放宽度」决定，而非由「是否错」决定。

六档语义（v2，写进论文的表格就用这六档）：
    1.0  严重放宽：禁令/停止 → 允许；限速上调 >= 30 km/h
    0.8  明显放宽：限速上调 10~29 km/h；警告 → 优先通行
    0.5  同族混淆：方向错、警告错，风险量级相当
    0.3  更严格：预测类比真实类更保守（把 80 读成 50），效率损失
    0.2  误刹车：通行类被读成停止/禁行/让行，不会超速但会幽灵刹车
    0.1  无害提示：被读成一般危险警告，只让车减速，几乎无危害

v1 把「任何标志 → 停车/警告」一律记 0。这条规则过宽：把限速 80 读成
停车标志会让车辆在高速路段幽灵急停，本身是追尾诱因，不该与「预测正确」
同权。v2 起 0 只留给对角线（预测正确），安全侧错误统一落在 0.1~0.3。

权重表以 CSV 落盘，便于领域人员逐条改分并复现（论文里要写这一点）。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------- 类别表
# GTSRB 官方 43 类（Stallkamp et al., 2012）
CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "End of speed limit 80", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No passing", 10: "No passing over 3.5 t",
    11: "Right-of-way at next intersection", 12: "Priority road",
    13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles over 3.5 t prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice or snow",
    31: "Wild animals crossing", 32: "End of all speed and passing limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing over 3.5 t",
}
NUM_CLASSES = 43

# 限速类 -> 该标志允许的最高车速（km/h）
# 6（解除 80 限速）与 32（解除全部限速与禁超）回到一般限速，取 100 / 130
SPEED_VALUE: Dict[int, int] = {
    0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80,
    6: 100, 7: 100, 8: 120, 32: 130,
}

SPEED_CLASSES = set(SPEED_VALUE)
WARNING_CLASSES = set(range(18, 32))          # 18-31 警告类
MANDATORY_CLASSES = set(range(33, 41))        # 33-40 指示类
PROHIBITIVE_CLASSES = {9, 10, 14, 15, 16, 17}  # 禁令/停止

# ------------------------------------------------ 非限速类的显式规则
# key = (真实类, 预测类)，value = 权重
# 只列需要偏离默认值的对，其余走 default_weight()
OVERRIDE: Dict[Tuple[int, int], float] = {
    # —— 停止 / 禁行被读成可通行：最严重
    (14, 13): 0.9,   # Stop -> Yield
    (14, 12): 1.0,   # Stop -> Priority road
    (17, 13): 1.0,   # No entry -> Yield
    (17, 12): 1.0,   # No entry -> Priority road
    (15, 12): 0.9,   # No vehicles -> Priority road
    (16, 12): 0.9,
    # —— 反向：被读成更严格的禁令，安全侧但有误刹车代价
    (13, 14): 0.2,   # Yield -> Stop：不会超速，但会无故急停
    (12, 14): 0.2,
    (12, 15): 0.2,
    # —— 禁超被读成解除禁超
    (9, 41): 1.0,    # No passing -> End of no passing
    (9, 32): 1.0,    # No passing -> End of all limits
    (10, 42): 1.0,
    (10, 32): 1.0,
    (9, 10): 0.4,    # 同类禁令互换
    (41, 9): 0.2,    # 解除 -> 禁止：安全侧，但会误判为不可超车
    # —— 优先权降级
    (12, 13): 0.5,   # Priority road -> Yield
    (11, 12): 0.6,
    # —— 指示方向错误
    (33, 34): 0.8,   # 右转 -> 左转：方向完全相反
    (34, 33): 0.8,
    (33, 35): 0.6,
    (34, 35): 0.6,
    (35, 33): 0.5,
    (35, 34): 0.5,
    # —— 警告类互换：都还在警告，风险量级接近
    (27, 28): 0.4,   # 行人 -> 儿童
    (28, 27): 0.4,
    (19, 20): 0.4,   # 左急弯 -> 右急弯
    (20, 19): 0.4,
}

# 安全侧低风险档：不再记 0，避免「幽灵急停」被当成无危害
W_SAFE_STOP, W_SAFE_HINT = 0.2, 0.1


def default_weight(t: int, p: int) -> float:
    """未在 OVERRIDE 中显式给出的 (t, p) 的兜底规则。"""
    if t == p:
        return 0.0
    # 限速 vs 限速：按放宽幅度
    if t in SPEED_CLASSES and p in SPEED_CLASSES:
        d = SPEED_VALUE[p] - SPEED_VALUE[t]
        if d >= 30:
            return 1.0
        if d >= 10:
            return 0.8
        if d <= -10:
            return 0.3
        return 0.4
    # 限速被读成禁令/停止：更严格，但会触发幽灵急停
    if p in PROHIBITIVE_CLASSES and t not in PROHIBITIVE_CLASSES:
        return W_SAFE_STOP
    # 禁令被读成限速：相当于放行
    if t in PROHIBITIVE_CLASSES and p in SPEED_CLASSES:
        return 0.9
    # 警告 -> 指示：丢失了危险提示
    if t in WARNING_CLASSES and p in MANDATORY_CLASSES:
        return 0.6
    # 指示 -> 警告：至少还在提示，偏安全
    if t in MANDATORY_CLASSES and p in WARNING_CLASSES:
        return 0.3
    # 所有 -> 停止 / 一般警告：低风险档，但不再是 0
    if p == 14:
        return W_SAFE_STOP
    if p == 18:
        return W_SAFE_HINT
    return 0.5


def build_matrix(overrides: Dict[Tuple[int, int], float] | None = None) -> List[List[float]]:
    """构造 43x43 权重矩阵 W[t][p]。对角线为 0（预测正确不计入）。"""
    W = [[0.0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            W[t][p] = default_weight(t, p)
    for (t, p), w in (overrides or OVERRIDE).items():
        W[t][p] = float(w)
    for i in range(NUM_CLASSES):
        W[i][i] = 0.0
    return W


# ---------------------------------------------------------------- 指标
def metrics_from_preds(true: Sequence[int], pred: Sequence[int],
                       W: List[List[float]] | None = None) -> Dict[str, float]:
    """由真实标签与预测标签直接算 ASR / SW-ASR / DMR。

    ASR   : 误分类比例（论文里的传统口径）
    SW-ASR: 加权误分类比例，值域 [0, 1]
    DMR   : w >= 0.8 的样本比例（「危险误判率」）
    """
    if W is None:
        W = build_matrix()
    n = len(true)
    if n == 0:
        raise ValueError("空输入")
    correct = 0
    wsum = 0.0
    danger = 0
    for t, p in zip(true, pred):
        if t == p:
            correct += 1
            continue
        w = W[int(t)][int(p)]
        wsum += w
        if w >= 0.8:
            danger += 1
    return {
        "n": n,
        "ASR": 1.0 - correct / n,
        "SW-ASR": wsum / n,
        "DMR": danger / n,
    }


def metrics_from_probs(true: Sequence[int], probs, W: List[List[float]] | None = None):
    """由概率矩阵（N x 43）算同一组指标，便于直接喂模型输出。"""
    pred = [int(max(range(len(row)), key=lambda k: row[k])) for row in probs]
    return metrics_from_preds(true, pred, W)


# ---------------------------------------------------------------- 落盘
def dump_csv(path: str, W: List[List[float]] | None = None) -> None:
    """把 43x43 矩阵导出为 CSV，供领域人员逐条校对。"""
    if W is None:
        W = build_matrix()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# SW-ASR 混淆权重矩阵 v2：W[真实类][预测类]，0=仅预测正确，1=最危险"])
        w.writerow(["# v2 相对 v1：安全侧错误（读成停车/禁行/警告）由 0 改为 0.1~0.3 低风险档"])
        w.writerow(["# 修改本文件后，用 load_csv() 覆盖默认规则，并在论文中注明评分人与一致性系数"])
        w.writerow(["true\\pred"] + [f"{i}:{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)])
        for t in range(NUM_CLASSES):
            w.writerow([f"{t}:{CLASS_NAMES[t]}"] + [f"{W[t][p]:.2f}" for p in range(NUM_CLASSES)])


def load_csv(path: str) -> List[List[float]]:
    W = [[0.0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    for r in rows[1:]:
        t = int(r[0].split(":")[0])
        for p, v in enumerate(r[1:]):
            W[t][p] = float(v)
    return W


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    W = build_matrix()

    def w(t, p):
        return W[t][p]

    print("== 关键对照（真实类 -> 预测类 : 权重）v2 ==")
    checks = [
        (1, 5, "限速30 -> 限速80", 1.0),
        (1, 2, "限速30 -> 限速50", 0.8),
        (5, 1, "限速80 -> 限速30", 0.3),
        (14, 12, "停止 -> 优先通行", 1.0),
        (13, 14, "让行 -> 停止", 0.2),
        (9, 41, "禁超 -> 解除禁超", 1.0),
        (27, 28, "行人 -> 儿童", 0.4),
        (26, 18, "信号灯 -> 一般警告", 0.1),
        (5, 14, "限速80 -> 停止", 0.2),
    ]
    ok = True
    for t, p, desc, want in checks:
        got = w(t, p)
        flag = "ok" if abs(got - want) < 1e-9 else "MISMATCH"
        if flag != "ok":
            ok = False
        print(f"  {desc:<22s} w={got:.2f}  期望 {want:.2f}  {flag}")

    print("\n== 同一批预测在三个指标下的区别 ==")
    # 场景 A：4 个样本全部攻破，但都是「把 80 读成 30」这种安全侧错误
    tA = [5, 5, 5, 5]
    pA = [1, 1, 1, 1]
    # 场景 B：同样 4 个全攻破，但都是「把 30 读成 80」
    tB = [1, 1, 1, 1]
    pB = [5, 5, 5, 5]
    for tag, t, p in [("A 安全侧错误", tA, pA), ("B 危险侧错误", tB, pB)]:
        m = metrics_from_preds(t, p, W)
        print(f"  {tag}: ASR={m['ASR']:.2f}  SW-ASR={m['SW-ASR']:.2f}  DMR={m['DMR']:.2f}")
    print("\n  说明：ASR 相同（都是 1.00），SW-ASR 与 DMR 把两者分开。"
          "\n        这正是传统 ASR 无法表达、而交通安全需要的信息。")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swasr_weights.csv")
    dump_csv(out, W)
    print(f"\n权重矩阵已导出：{out}")
    print("自检通过" if ok else "自检未通过")
