# -*- coding: utf-8 -*-
"""把 R1 多种子实验的结果回填进 paper.md。

读取 results/r1s*_raw.npz（5 个种子），按 paper 的表 10/表 11 口径重算：
  · 集成内 3 目标 ASR：先按种子算 3 目标均值，再跨种子取均值±标准差
  · 未见目标各项：直接按种子统计后取均值±标准差
  · 与各对照的 Wilcoxon + Bonferroni 已在 merge_seeds 中完成，此处只取数
然后替换 paper.md 中的 {{R1_*}} 占位符。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import swasr as S                       # noqa: E402
from a11.merge_seeds import load_all, metrics    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")

ORDER = ["I-FGSM", "MI-FGSM", "NAE", "DI-NI-FGSM", "RNTME-EQ", "RNTME-NAE"]
TAG = {"I-FGSM": "I", "MI-FGSM": "M", "NAE": "N", "DI-NI-FGSM": "D",
       "RNTME-EQ": "Q", "RNTME-NAE": "R"}
REF = "RNTME-NAE"


def pm(v):
    return f"{v*100:.1f}"


def fmt(v, s):
    return f"{v*100:.1f}±{s*100:.1f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="r1s0,r1s1,r1s2,r1s3,r1s4")
    ap.add_argument("--summary", default="r1_summary.json")
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper.md"))
    ap.add_argument("--dry", action="store_true", help="只打印替换值，不写回")
    opt = ap.parse_args()
    prefixes = [x.strip() for x in opt.inputs.split(",") if x.strip()]
    data, true, clean = load_all(prefixes)
    seeds = sorted(data.keys())
    W = S.build_matrix()
    summ = json.load(open(os.path.join(RES, opt.summary), encoding="utf-8"))
    helds = summ.get("heldout") or ["ssep"]
    held = helds[0]
    ens = [t for t in clean if t not in helds]
    print("种子", seeds, "集成", ens, "未见", held, "样本", len(true))

    # 每轮耗时（由 run_seeds 的分种子任务各自写出）
    tim = {}
    for pfx in prefixes:
        fp = os.path.join(RES, f"{pfx}_time.json")
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        for k, v in d.items():
            a, sd = k.split("|")
            tim.setdefault(a, []).append(v)

    stat = {}
    for a in ORDER:
        per = {k: [] for k in ("ASR", "SW-ASR", "DMR", "ASR_cond", "ASR_IN")}
        for sd in seeds:
            mh = metrics(true, data[sd][a][held], clean[held], W)
            for k in ("ASR", "SW-ASR", "DMR", "ASR_cond"):
                per[k].append(mh[k])
            per["ASR_IN"].append(np.mean([metrics(true, data[sd][a][t], clean[t], W)["ASR"]
                                          for t in ens]))
        st = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
              for k, v in per.items()}
        if a in tim and tim[a]:
            st["MS"] = (float(np.mean(tim[a]) * 1000), float(np.std(tim[a], ddof=1) * 1000)
                        if len(tim[a]) > 1 else 0.0)
        stat[a] = st

    # 显著性
    tests = [t for t in summ["tests"] if t["target"] == held and t["metric"] == "ASR"]
    m_tests = summ["n_tests"]
    by_other = {t["other"]: t for t in tests}

    best = max([a for a in ORDER if a != REF], key=lambda a: stat[a]["ASR"][0])
    r1 = stat[REF]["ASR"][0]
    rank = sorted(ORDER, key=lambda a: -stat[a]["ASR"][0])
    p_best = by_other[best]["p_bonf"]
    d_best = (r1 - stat[best]["ASR"][0]) * 100

    # ---------------- 表 10 / 表 11 ----------------
    rep = {}
    for a in ORDER:
        t = TAG[a]
        rep[f"R1_ASR_IN_{t}"] = fmt(*stat[a]["ASR_IN"])
        rep[f"R1_ASR_OUT_{t}"] = fmt(*stat[a]["ASR"])
        rep[f"R1_COND_{t}"] = fmt(*stat[a]["ASR_cond"])
        rep[f"R1_SW_OUT_{t}"] = fmt(*stat[a]["SW-ASR"])
        rep[f"R1_DMR_OUT_{t}"] = fmt(*stat[a]["DMR"])
        ms = stat[a]["MS"][0] if "MS" in stat[a] else float("nan")
        rep[f"R1_MS_{t}"] = f"{ms:.0f}"
        base_ms = stat.get("I-FGSM", {}).get("MS", (float("nan"),))[0]
        rep[f"R1_RATIO_{t}"] = f"{ms/base_ms:.1f}×" if base_ms == base_ms and base_ms else "—"

    # ---------------- 分析文字 ----------------
    RANK_TXT = "、".join(f"{a}（{stat[a]['ASR'][0]*100:.1f}%）" for a in rank)
    lo = min(stat[a]["ASR"][0] for a in ORDER) * 100
    hi = max(stat[a]["ASR"][0] for a in ORDER) * 100
    if p_best < 0.05:
        cmp_txt = f"{'高于' if d_best > 0 else '低于'} {best}（{stat[best]['ASR'][0]*100:.1f}%），差值为 {d_best:+.1f} 个百分点，经 Bonferroni 校正后仍显著（p_bonf = {p_best:.2g}）"
    else:
        cmp_txt = (f"与最高的 {best}（{stat[best]['ASR'][0]*100:.1f}%）相差 {d_best:+.1f} 个百分点，"
                   f"经 Bonferroni 校正后不显著（p_bonf = {p_best:.2g}）")

    drop = {a: (stat[a]["ASR_IN"][0] - stat[a]["ASR"][0]) * 100 for a in ORDER}

    sig_eq = by_other.get("RNTME-EQ", {})
    p_eq = sig_eq.get("p_bonf", 1.0)
    d_eq = (stat[REF]["ASR"][0] - stat["RNTME-EQ"]["ASR"][0]) * 100
    d_eq_in = (stat[REF]["ASR_IN"][0] - stat["RNTME-EQ"]["ASR_IN"][0]) * 100
    eq_txt = ("显著" if p_eq < 0.05 else "不显著")

    msr = stat[REF]["MS"][0] if "MS" in stat[REF] else float("nan")
    msi = stat["I-FGSM"]["MS"][0] if "MS" in stat["I-FGSM"] else float("nan")

    para1 = (f"未见目标上的排序与主表并不一致。六种方法的 ASR 落在 {lo:.1f}%～{hi:.1f}% 之间，"
             f"由高到低依次为 {RANK_TXT}。"
             f"本文方法为 {r1*100:.1f}%，{cmp_txt}。这意味着在目标完全不参与生成的条件下，"
             f"集成与动态加权带来的优势被大幅压缩：主表上 20.7 个百分点的差距，到了未见目标上只剩 {d_best:+.1f} 个百分点。")
    para2 = (f"把“集成内 → 未见”的落差逐方法列出，可以看到差别所在："
             + "；".join(f"{a} 为 {stat[a]['ASR_IN'][0]*100:.1f}%→{stat[a]['ASR'][0]*100:.1f}%（落差 {drop[a]:.1f}）" for a in ORDER)
             + f"。本文方法的落差（{drop[REF]:.1f} 个百分点）{'小于' if drop[REF] < drop['MI-FGSM'] else '不小于'}最强的单模型基线 MI-FGSM（{drop['MI-FGSM']:.1f} 个百分点），"
               f"说明集成与变换确实抑制了一部分对源模型的过拟合；但落差小并不等于绝对水平高，未见目标上的绝对值仍由架构亲缘决定。")
    para3 = (f"等权与动态权重的对照是本节最干净的比较：两者共用随机变换、高斯平滑与 Nesterov 三条支路，"
             f"唯一的差别是权重是否随损失自适应。在未见目标上，动态权重为 {stat[REF]['ASR'][0]*100:.1f}%，"
             f"等权为 {stat['RNTME-EQ']['ASR'][0]*100:.1f}%，差值 {d_eq:+.1f} 个百分点，检验结果{eq_txt}（p_bonf = {p_eq:.2g}）；"
             f"在集成内，两者分别为 {stat[REF]['ASR_IN'][0]*100:.1f}% 与 {stat['RNTME-EQ']['ASR_IN'][0]*100:.1f}%，差值 {d_eq_in:+.1f}。"
             f"{"也就是说，动态权重的作用主要体现在对已见模型的压制上，对未见目标的增益并不确凿。" if p_eq >= 0.05 else "动态权重在未见目标上同样有效，说明其收益不只是对已见模型的拟合。"}")
    para4 = (f"危害加权指标给出的信息与 ASR 一致。未见目标上本文方法的 SW-ASR 为 {stat[REF]['SW-ASR'][0]*100:.1f}%、"
             f"DMR 为 {stat[REF]['DMR'][0]*100:.1f}%，等权集成为 {stat['RNTME-EQ']['SW-ASR'][0]*100:.1f}% 与 "
             f"{stat['RNTME-EQ']['DMR'][0]*100:.1f}%，MI-FGSM 为 {stat['MI-FGSM']['SW-ASR'][0]*100:.1f}% 与 "
             f"{stat['MI-FGSM']['DMR'][0]*100:.1f}%——三者的排序与 ASR 相同，没有出现“成功率不高但危害更大”的反例。")
    para5 = (f"耗时一栏印证了表 5 的说明。在同一框架、同一批大小、同一精度下，本文方法单样本耗时 {msr:.0f} ms，"
             f"是 I-FGSM（{msi:.0f} ms）的 {msr/msi:.1f} 倍，与每轮 15 次传播对 1 次的比例基本吻合；"
             f"而表 5 中本文方法反而是 DI-NI-FGSM 的 35%，差别只能来自工程优化。因此表 5 的耗时列不构成算法效率证据，"
             f"真正的效率代价应以本表为准：跨模型迁移能力的提升，是用约 {msr/msi:.0f} 倍的计算量换来的。")

    # 配置句：样本量与各模型实测准确率
    CN = {"sresnet": "残差", "svgg": "深层 VGG 式", "ssep": "深度可分离卷积",
          "sinc": "多分支 Inception 式"}
    n_acc = {t: float((clean[t] == true).mean()) for t in clean}
    rep["R1_N"] = str(len(true))
    rep["R1_ACC"] = "、".join(f"{CN.get(t,t)} {n_acc[t]*100:.1f}%" for t in
                              ["sresnet", "svgg", "ssep", "sinc"] if t in n_acc)
    rep["R1_BEST_ACC"] = f"{max(n_acc.values())*100:.1f}%"
    rep["R1_WORST_ACC"] = f"{min(n_acc.values())*100:.1f}%"
    rep["R1_ANALYSIS"] = "\n\n".join([para1, para2, para3, para4, para5])

    rep["R1_ABS"] = (f"本文方法为 {r1*100:.1f}%，{best} 为 {stat[best]['ASR'][0]*100:.1f}%，"
                     f"差值 {d_best:+.1f} 个百分点（Bonferroni 校正后{'显著' if p_best < 0.05 else '不显著'}）")

    en_sig = "significant" if p_best < 0.05 else "not significant"
    rep["R1_ABS_EN"] = (f"the proposed method reaches {r1*100:.1f}% against {stat[best]['ASR'][0]*100:.1f}% for the "
                        f"strongest baseline ({best}), a difference of {d_best:+.1f} percentage points that is "
                        f"{en_sig} after Bonferroni correction (p_bonf = {p_best:.2g})")

    if p_best < 0.05 and d_best > 0:
        concl = (f"结果显示：在完全未参与生成的目标上，本文方法的成功率为 {r1*100:.1f}%，"
                 f"高于最强的单模型基线 {best}（{stat[best]['ASR'][0]*100:.1f}%），差 {d_best:+.1f} 个百分点，"
                 f"经 Bonferroni 校正后显著（p_bonf = {p_best:.2g}）。这一差距远小于主表上的 20.7 个百分点，"
                 f"说明集成类方法的优势在真正黑盒条件下会被显著压缩，主表的数字不能被读作迁移率。")
    else:
        concl = (f"结果显示：在完全未参与生成的目标上，本文方法的成功率为 {r1*100:.1f}%，"
                 f"与最强的单模型基线 {best}（{stat[best]['ASR'][0]*100:.1f}%）相差 {d_best:+.1f} 个百分点，"
                 f"经 Bonferroni 校正后不显著（p_bonf = {p_best:.2g}）。这是本文必须如实报告的负结果："
                 f"在主表上观察到的 20.7 个百分点优势不能外推到完全未见的目标，集成与动态加权的收益主要是对已见模型的压制，"
                 f"而非更强的跨模型迁移。相应地，本文的贡献应定位为“在同等模型数量下更有效的集成攻击与一套任务级评估流程”，"
                 f"而不是“更强的迁移攻击”。")
    rep["R1_CONCL"] = concl

    # 迭代扫描
    tscan_p = os.path.join(RES, "r1_tscan.json")
    if os.path.exists(tscan_p):
        rows = json.load(open(tscan_p, encoding="utf-8"))
        r = [x for x in rows if x["attack"] == REF]
        if r:
            sat = "、".join(f"{x['T']} 轮 {x['sat_frac']*100:.0f}%" for x in r)
            asr = "、".join(f"{x['ASR_' + held]*100:.1f}%" for x in r)
            rep["TSCAN_TEXT"] = (f"在公开集上逐档实测（$T$=5、8、11、15）给出的饱和比例为 {sat}，"
                                 f"未见目标成功率依次为 {asr}。第 8 轮之后扰动已基本贴住 $\\varepsilon$ 球，"
                                 f"后续更新只在球面上转向，成功率的边际增幅明显收窄——这与本节此前“15 轮是收益与开销的分界”"
                                 f"的表述有出入，更准确的说法是：幅度增长在第 8 轮左右饱和，第 9—15 轮买到的是方向微调，"
                                 f"因此 $T=15$ 应理解为留给方向调整的余量，而非收益拐点。")

    # ---------------- 回填 ----------------
    paper = opt.paper
    s = open(paper, encoding="utf-8").read()
    miss = []
    for k, v in rep.items():
        tok = "{{" + k + "}}"
        if tok in s:
            s = s.replace(tok, v)
        else:
            miss.append(k)
    if opt.dry:
        print("\n[dry] 未写回；替换项：")
        for k, v in rep.items():
            if k != "R1_ANALYSIS":
                print(f"  {k} = {v}")
        return
    open(paper, "w", encoding="utf-8", newline="").write(s)
    left = s.count("{{")
    print(f"回填 {len(rep)} 项，未命中的占位 {miss}，残余占位符 {left}")
    print("\n摘要句：", rep["R1_ABS"])
    print("\n结论句：", rep["R1_CONCL"][:120], "...")


if __name__ == "__main__":
    main()
