# A1.1 交通标志识别任务级攻击实验（GTSRB）

对应《T-ITS 投稿补充方案》中的 **A1.1**。目的不是再报一个更高的 ASR，而是回答交通工程读者关心的问题：
**扰动把限速标志改成什么，改完之后车会不会超速。**

## 目录

| 文件 | 作用 |
|---|---|
| `swasr.py` | SW-ASR / DMR 指标与 43×43 混淆权重矩阵；纯 Python，可独立运行自检 |
| `swasr_weights.csv` | 权重矩阵导出，交给领域人员逐条校对（论文需注明评分人） |
| `data.py` | GTSRB 解析、分层划分、按类别抽样、ROI 裁剪 |
| `models.py` | 两套模型：`small`（4 个架构互异的小网，32 px，CPU 可跑）／`full`（5 替代 + 2 未见，torchvision 主干） |
| `attacks.py` | I-FGSM / NAE / DI-NI-FGSM / RNTME-NAE |
| `train.py` | 训练替代模型与目标模型 |
| `eval_attack.py` | 生成对抗样本、按目标统计、输出 CSV/JSON |

## 指标定义

设真实类 $t$、预测类 $p$，权重矩阵 $W[t][p]\in[0,1]$ 表示这次误判的**危险程度**：

- `ASR` = 误分类比例（传统口径）
- `SW-ASR` = $\frac{1}{N}\sum_i W[t_i][p_i]$，加权成功率
- `DMR` = $\frac{1}{N}\sum_i \mathbb{1}[W[t_i][p_i]\ge 0.8]$，危险误判率

权重只惩罚**朝更宽松方向**的错。把限速 80 读成 30 是效率损失（w=0.3），把 30 读成 80 是违法与事故（w=1.0）。
六档语义（v2）：`1.0` 严重放宽 / `0.8` 明显放宽 / `0.5` 同族混淆 / `0.3` 更严格（效率损失）/
`0.2` 误刹车（读成停止或禁行，不超速但会幽灵急停）/ `0.1` 仅减速（读成一般危险警告）。

v1 曾把「任何标志 → 停车/警告」一律记 0。这条规则过宽：限速 80 读成停车标志会让车在高速路段
无故急停，是追尾诱因。v2 起 0 只留给对角线（预测正确），安全侧错误落在 0.1~0.3。改动使 SW-ASR
绝对值上移 0.01~0.02，四种攻击的相对次序与结论不变。旧结果备份在 `results/pilot_v1.csv`，
用 `python -m a11.recompute_v2 --prefix pilot` 可从逐样本 npz 重算。

自检里给了个最小演示：两批样本 ASR 都是 1.00，但 SW-ASR 分别是 0.30 与 1.00，DMR 分别是 0.00 与 1.00。
**这就是 ASR 表达不了、而交通安全必须知道的那部分信息。**

## 跑法

### 全量（GPU，论文正式数据）

```bash
# 1) 训练 5 个替代模型 + 2 个未见目标（224 px）
python -m a11.train --arch full --size 224 --epochs 30 --bs 64

# 2) 攻击评测（源 ResNet50，集成 5 模型，未见目标 ResNet101 + DenseNet121）
python -m a11.eval_attack --size 224 --n 5000 --T 15 \
  --ensemble resnet50,efficientnet_b0,vgg19,mobilenet_v2,googlenet \
  --heldout resnet101,densenet121 --sigma 3 --out full
```

### CPU 试点（本机无独显时验证链路）

```bash
python -m a11.train --arch all --size 32 --epochs 10 --train-subset 12000
python -m a11.eval_attack --size 32 --n 400 --T 15 \
  --ensemble sresnet,svgg,ssep --heldout sinc --sigma 1.5 --out pilot
```

试点与正式的差别（论文里不能混用）：32 px 而非 224 px、4 个小网而非 torchvision 主干、
训练子集 12 000 幅、M=3 集成 / 1 个未见目标。**试点只用于验证代码与指标行为，不用于论文数据。**

## 论文里要交代的三件事

1. **权重是谁定的。** 现表是规则推导的初版。正式稿应写：「由 3 名交通工程/交通安全方向人员独立打分，
   取均值，Kendall's W = X.XX 表明一致性可接受」。把 `swasr_weights.csv` 发给他们改，改完用 `--weights` 传入。
2. **早停粒度是 batch 不是样本。** `attacks.py` 的早停一旦 batch 内全部样本攻破即停，这是工程近似。
3. **污染隔离。** `--heldout` 里的模型绝不参与 `--ensemble`，否则又回到原稿「目标集就是集成集」的老问题。

## 已知限制

- ROI 用固定裁掉 10% 边距近似，未用 GT 里的逐图 Roi 坐标（那需要另读 CSV，对 32 px 影响很小，224 px 建议补上）。
- `rand_transform` 是论文 3.3 节六类变换的简化实现（缩放+平移+亮度+对比度+模糊），缺旋转与 JPEG 模拟。
- 单一 $\varepsilon=16/255$ 单步长，未做 $\varepsilon$ 扫描。
