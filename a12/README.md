# A1.2 目标检测任务级攻击实验（KITTI / 合成场景）

> **⚠️ 先训练权重**：本仓库的 `a12/chk/` 是**空目录**，没有随包发布任何 `*.pt`
> （其中 Faster R-CNN 单独就有 73 MB，超过 GitHub 能优雅处理的上限）。
> 跑本模块的任何命令前，请先执行下面的三个训练命令重建权重。

> **需要 torchvision**：本模块是仓库里唯一依赖 `torchvision` 的部分
> （检测器定义与 VOC 自动下载）。a11 / a13 不需要。

对应《T-ITS 投稿补充方案》中的 **A1.2**。A1.1 回答「标志识别错了会不会超速」，
A1.2 回答另一个问题：**扰动能不能让车看不见人和车**。

分类攻击的成功率不能直接外推到检测：检测器要同时做定位与分类，输出是框的集合，
而且中间有 NMS 与 RPN，扰动经过的结构完全不同。所以必须单独做一遍。

## 目录

| 文件 | 作用 |
|---|---|
| `dataset.py` | VOC / KITTI 适配，标注统一映射到 COCO id（person/bicycle/car/motorcycle/bus） |
| `synth.py` | 合成交通标志检测场景：GTSRB 精确 ROI 贴图 + 程序生成的天空/路面/车道线 |
| `detectors.py` | torchvision 检测器封装（SSDLite / FasterRCNN-MobileNet / FasterRCNN-ResNet50 / RetinaNet），含改头与微调权重的加载 |
| `attacks.py` | I-FGSM / NAE / DI-NI-FGSM / RNTME-NAE 的**检测版** |
| `metrics.py` | mAP@0.5（all-point AP）、漏检率、检测数变化；纯 numpy，无 pycocotools 依赖 |
| `train.py` | 在合成场景上微调检测器（试点专用；KITTI 正式实验不需要） |
| `run.py` | 主实验：干净基线 → 生成扰动 → 集成内/未见检测器评估 → CSV/JSON/npz |

## 指标

- **mAP@0.5**：单 IoU 阈值、按置信度贪心匹配、all-point 插值（VOC2007 主指标口径），
  difficult 框不计 FP。
- **漏检率 MR**：`1 - recall`，recall 取「置信度 ≥ 0.5 且 IoU ≥ 0.5 的 GT 被检出比例」。
  正式实验报**行人漏检率**（KITTI `Pedestrian`），试点报**限速标志漏检率**（合成场景没有行人）。
- **检测数比** `ndet_adv / ndet_clean`：框总量变化，反映「目标消失」而非「换个框」。

## 检测版攻击与分类版的三个差别（论文要写）

1. **损失是检测损失**：分类损失 + 框回归损失之和，权重由各检测器自己定
   （Faster R-CNN 为 1:1，SSD 为 hard-negative mining 后的分类损失 + smooth-L1）。
2. **几何变换必须同步变换 GT 框**。分类任务旋转图像后标签不变，检测任务旋转图像
   却不旋转框，loss 就与图像对不上。`attacks.warp_boxes()` 按
   `o = s·i + (1-s)·W/2 - s·t·W/2` 把框搬到变换后的坐标系。
   消融开关：`--warp-boxes 0`。

   > 实测结论（n=25，T=5，B=3）：同步 79.4% / 不同步 78.4%（ssdlite_a），
   > 同步 68.9% / 不同步 69.8%（ssdlite_b），差异在单种子噪声内。
   > 即**框同步是实现正确性问题，不是当前设定下的效果来源**——缩放只有 0.80~1.00、
   > 位移上限 6%，错位幅度小；且分类分支对轻微错位不敏感。要把它做成卖点，
   > 需把缩放下界压到 0.5、位移放到 15% 再验证。
3. **损失尺度跨模型不可比**。检测器总损失从 2 到 10 不等，直接取倒数会让损失大的
   模型独占权重。动态权重改用**相对损失** `L_i / L_i^(0)`（首步值）归一化后再取倒数，
   温度、EMA、L1 归一化、高斯平滑与论文式(7)–(10) 一致。

## 跑法

### 试点（CPU，合成场景）

```bash
# 1) 训练检测器（集成 2 个 + 未见 1 个）
python -m a12.train --arch ssdlite      --n 400 --epochs 5 --seed 0 --out a12/chk/ssdlite_a.pt
python -m a12.train --arch ssdlite      --n 400 --epochs 5 --seed 1 --out a12/chk/ssdlite_b.pt
python -m a12.train --arch frcnn_mnv3   --n 400 --epochs 5 --seed 0 --out a12/chk/frcnn_mnv3.pt

# 2) 攻击评测
python -m a12.run --data synth --n 80 --T 10 --B 3 \
    --ens ssdlite_a,ssdlite_b --heldout frcnn_mnv3 \
    --ckpt ssdlite_a=a12/chk/ssdlite_a.pt,ssdlite_b=a12/chk/ssdlite_b.pt,frcnn_mnv3=a12/chk/frcnn_mnv3.pt \
    --tag pilot
```

### 正式（GPU，KITTI 零样本）

KITTI 的行人/车辆本来就是 COCO 类，**不需要微调**，直接用 COCO 预训练检测器评估：

```bash
# 数据放到 data/kitti/training/{image_2,label_2}
python -m a12.run --data kitti --root data --n 1000 --size 384x1280 --T 20 --B 5 \
    --ens ssdlite,frcnn_mnv3,retina_r50 --heldout frcnn_r50 --mr-class 1 --tag full
```

## 试点结果（合成场景，不可进论文）

配置：384×512，60 张测试图 / 126 个目标，T=8，B=3，ε=8/255，集成 `ssdlite_a`+`ssdlite_b`，
未见 `frcnn_mnv3`（干净 mAP 0.720 / 0.764 / 0.879）。

| 攻击 | ssdlite_a（白盒） | ssdlite_b（集成内） | frcnn_mnv3（未见） |
|---|---|---|---|
| I-FGSM | 96.9% | 20.0% | 7.7% |
| NAE | 96.5% | 40.2% | 15.8% |
| DI-NI-FGSM | 53.0% | 15.3% | 11.8% |
| RNTME-NAE | 89.6% | **86.7%** | **18.3%** |

（mAP@0.5 相对下降）

三个可写进论文的观察：

1. **基线是单点突破**：I-FGSM 在白盒成员上掉 96.9%，在同一个集成的另一成员上只掉 20.0%，
   5 倍落差；RNTME 是 89.6% / 86.7%，基本持平。集成攻击该看「对多个成员是否都有效」。
2. **未见目标上的优势真实但有限**：RNTME 18.3% 最高，是 I-FGSM 的 2.4 倍，远小于集成内的差距。
3. **检测任务比分类更难迁移**：同为 RNTME，A1.1 分类未见目标 ASR 34.3%，这里只有 18.3%。
   RPN 候选框采样 + NMS 后处理 + 单/两阶段架构差异构成更强的壁垒。

## 为什么试点用合成场景

本机到 KITTI（12 GB）、VOC（460 MB）、COCO（403 MB/分片）的镜像实测都在 25 KB/s 以下，
只有 `download.pytorch.org`（模型权重）可达。因此试点用 GTSRB 现成标注合成带框场景：
背景是程序生成的天空/路面/车道线，前景是按精确 ROI 裁出的真实标志贴图。

这样得到的**绝对精度不能写进论文**，它的用途有三个：

1. 验证整条链路正确（尤其是框同步这一条）；
2. 做「框同步 / 不同步」的消融 —— 这是实现正确性问题，与数据真实度无关；
3. 观测「分类攻击搬到检测任务上会衰减多少」的**量级** —— 这个结论与数据来源无关。

## 已知限制

- 合成场景只有 5 粗类、无遮挡、无运动模糊，分布与真实道路差距大；
- 集成里两个 SSDLite 只有初始化种子不同，多样性不足，会低估集成攻击的迁移性；
- 单种子单次运行，正式实验需 5 个种子 + Wilcoxon 检验；
- 检测器训练只用了 400 张合成图，其干净 mAP 明显低于真实场景的检测器。
