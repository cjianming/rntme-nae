# a13 — 对抗样本的防御评估

回答两个审稿人必问的问题：链路预处理能不能削弱扰动？模型级防御值不值得付准确率代价？

## 三个模块

| 文件 | 作用 |
|---|---|
| `defenses.py` | 10 种输入级防御 + `Defended` 包装（含 BPDA 直通） |
| `run.py` | 生成对抗样本 → 自适应样本 → 评估矩阵 |
| `adv_train.py` | PGD 对抗训练，回答「模型级防御抬多高门槛」 |

## 跑法

```bash
# 1) 生成（复用 A1.1 的攻击实现，同配置：32px / T=15 / B=5 / eps=16/255）
python -m a13.run --stage gen --n 600 --T 15 --B 5

# 2) 自适应攻击（BPDA）：攻击者知道防御存在
python -m a13.run --stage gen --n 600 --T 15 --B 5 \
    --bpda jpeg75,jpeg50,bit4,blur10 --attacks RNTME-NAE,DI-NI-FGSM

# 3) 评估
python -m a13.run --stage eval --n 600 --bpda jpeg75,jpeg50,bit4,blur10

# 4) 对抗训练
python -m a13.adv_train --arch sresnet --epochs 5 --eps 8/255

# 5) 出图与报告
python -m a13.make_figures && python -m a13.make_report
```

## 四条不能省的评估纪律

1. **成对报**。任何防御都要同时报「ASR」和「干净准确率」。只看 ASR 下降，
   等于把模型打瞎了还叫防御有效。`ASR_rel` 是扣掉自身代价后的相对攻击率，
   `net_gain = ΔASR − Δacc` 是净收益，≤0 表示这条防御本质只是把模型弄笨。

2. **必须做自适应攻击**。非自适应评估只能说明「这类扰动被削弱」。判据是：
   攻击者知道防御并针对它优化后会怎样。实现是 BPDA —— 前向真过 JPEG/量化/模糊，
   反向把防御当恒等（`xd = x + (f(x) - x).detach()`）。

3. **随机化防御要多次采样投票**。`randpad` 单次随机会把随机性算成准确率损失
   （32 px 试点里 ssep 掉了 16 pp），评估时取 5 次多数。攻击侧若要对它做自适应，
   应该用 EOT 而不是 BPDA（本脚本未做，留正式实验）。

4. **报 SW-ASR 而不只是 ASR**。防御后残留的误判里，危险方向占多少，
   比总量更有意义 —— 这直接复用 `a11/swasr.py` 的权重表。

## 尺寸效应（重要）

32 px 上一个 8×8 JPEG 块占整幅图 6.25%，224 px 上只占 0.13%，**相对影响差 49 倍**。
所以本目录的 JPEG 结论不能外推到正式分辨率，224 px 必须重跑。

## 结果文件

- `results/a3_plain_<攻击>.npz`：对抗图像（uint8）、标签、L2、干净预测
- `results/a3_bpda_<防御>_<攻击>.npz`：自适应攻击的对抗图像
- `results/a3_defense.csv`：全矩阵（kind × defense × attack × target）
- `results/a3_meta.json`：配置与各防御下的干净准确率
- `results/advtrain_summary.json`：对抗训练的干净/迁移/自适应三组数

## 对抗训练：三个坑（都踩过）

1. **学习率要比干净训练小**。lr=0.05（与 A1.1 干净训练相同）首轮干净准确率只有 28.2%，
   降到 0.02 曲线才稳。对抗样本让梯度方向更抖，同步长会把权重推过头。
2. **别用「先用干净数据校准再冻结 BN」来省时间**。那份 running stats 来自未训练的随机权重，
   与训练后的权重完全不匹配，结果更差（5.2%）。BN 必须随权重一起收敛，这也是它费轮数的原因。
3. **收敛慢约 2.4 倍**。60 步时干净训练到 46.5%、对抗训练只到 19.5%（两条都在升，说明实现没错）。
   按 A1.1 的 5 轮 95% 外推，要 12~15 轮才够，文献标准配置是 30 轮以上。CPU 上做不完，
   这决定了对抗训练只能留给 GPU 正式实验。

## 踩坑

- `bit_depth` 量化后必须夹回 0..255：255 在 q=16 时 round 到 256，还原后越界成 1.0078。
- `dehaze` 的 `A` 要 reshape 成 `(B,C,1,1)`，用 `view(-1,1,1,1)` 会变成 `(B*C,1,1,1)` 后广播错维。
- `GTSRB.__getitem__` 直接索引返回 numpy（DataLoader 才转 tensor），自检里要手动 `torch.from_numpy`。
- 沙箱里 `rm -f` 会挂起，清理临时文件用 Python 的 `os.remove`。
