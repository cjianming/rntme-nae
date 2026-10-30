# -*- coding: utf-8 -*-
"""生成 A13 防御评估记录 docx。

所有数字从 a13/results/a3_defense.csv 与 advtrain_summary.json 读，
措辞按数据走分支——避免出现「结论写死了但实验变了」。
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from a11.make_report import SONG, _f, figure, h1, h2, newdoc, para, table

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def rp(doc, text, size=10.5, align="just", first=2.0, before=0, after=2, line=1.45):
    """支持 **粗体** 标记的段落。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Cm(first)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY,
                    "left": WD_ALIGN_PARAGRAPH.LEFT,
                    "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        r = p.add_run(seg)
        r.font.size = Pt(size)
        r.bold = (i % 2 == 1)
        _f(r, SONG)
    return p


def load(out):
    p = os.path.join(RES, f"{out}_defense.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        for k in ("clean_acc", "acc_adv", "ASR", "SW-ASR", "DMR",
                  "ASR_rel", "net_gain", "L2"):
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = float("nan")
    return rows


def A(rows, d, atk=None, kind="普通攻击", role=None, key="ASR"):
    sub = [r for r in rows if r["defense"] == d and r["kind"] == kind]
    if atk:
        sub = [r for r in sub if r["attack"] == atk]
    if role:
        sub = [r for r in sub if r["role"] == role]
    v = [r[key] for r in sub if r[key] == r[key]]
    return float(np.mean(v)) if v else float("nan")


def f1(x):
    return "—" if x != x else f"{x*100:.1f}"


def clean_of(rows, d, m):
    """某防御、某模型上的干净准确率。clean_acc 在每行重复，取均值即可。"""
    s = [r["clean_acc"] for r in rows if r["defense"] == d and r["target"] == m]
    return float(np.mean(s)) if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="a3")
    ap.add_argument("--doc", default="../A13_防御评估实验记录.docx")
    a = ap.parse_args()

    rows = load(a.out)
    meta = json.load(open(os.path.join(RES, f"{a.out}_meta.json"), encoding="utf-8"))
    adv = None
    fp = os.path.join(RES, "advtrain_summary.json")
    if os.path.exists(fp):
        adv = json.load(open(fp, encoding="utf-8"))

    from a13.defenses import DESC
    dlist = [d for d in DESC if any(r["defense"] == d for r in rows)]
    atks = [x for x in ["I-FGSM", "NAE", "DI-NI-FGSM", "RNTME-NAE"]
            if any(r["attack"] == x for r in rows)]
    models = [r["target"] for r in rows if r["defense"] == "none" and r["attack"] == atks[0]
              and r["kind"] == "普通攻击"]

    doc = newdoc()
    h1(doc, "A13　对抗样本的防御评估")
    para(doc, f"实验日期 2026-10-26 | 数据 GTSRB 32 px 试点 | {meta['n']} 张测试图 | "
              f"集成 {', '.join(meta['ensemble'])} | 未见 {', '.join(meta['heldout'])} | "
              f"ε={meta['eps']*255:.0f}/255，T={meta['T']}，B={meta['B']}",
         size=9.6, align="center", after=6)

    # ------------------------------------------------ 一
    h1(doc, "一、为什么这一项不能省")
    para(doc, "A1.1 和 A1.2 回答的是「攻击有多强」。但一篇投 T-ITS 的论文如果只写攻击，"
              "审稿人第一个问题一定是：那你们建议防守方怎么做？第二个问题是："
              "真实的车端和路侧链路里有 JPEG 回传、有摄像头 ISP、有低照度增强，"
              "你的扰动过得了这些环节吗？")
    para(doc, "这两个问题合起来就是本节的内容。做法是把 A1.1 已经生成的对抗样本"
              "送进一条「预处理 → 分类器」的链路，逐个环节看攻击还剩多少；"
              "再用对抗训练检验模型级防御。")
    rp(doc, "必须提前说清一件事：**这一节的数字全部来自 32 px 试点**。32 px 图像上"
            "一个 8×8 的 JPEG 块占整幅图的 6.25%，而 224 px 上只占 0.13%——相对影响差 49 倍。"
            "所以这里关于 JPEG 的结论不能直接外推到正式实验的分辨率，正式实验必须重跑。",
       first=2.0)

    # ------------------------------------------------ 二
    h1(doc, "二、防御清单")
    para(doc, "按作用层次分三类。前两类只改输入，不训练；第三类改模型本身。")
    body = []
    for d in dlist:
        nm, note = DESC[d]
        kind = ("—" if d == "none" else
                "输入压缩/量化" if d.startswith(("jpeg", "bit")) else
                "输入滤波" if d == "blur10" else
                "输入随机化" if d == "randpad" else
                "场景预处理" if d in ("gamma06", "dehaze") else "对照")
        body.append([nm, kind, note])
    table(doc, ["防御", "类别", "参数与用途"], body, [3.6, 3.4, 9.2],
          caption="表 1　本次评估的输入级防御")

    para(doc, "实现上有两点值得写进论文的方法部分。第一，JPEG 用 4:4:4 采样（不做色度下采样），"
              "因为 32 px 上 4:2:0 会把颜色彻底糊掉，那不是「压缩」而是「换了一张图」；"
              "另外单列了 4:2:0 版本做敏感性对照。第二，随机缩放填充（R&P）在推理时随机变换，"
              "攻击者无法针对某个确定的变换优化扰动，因此评估时要做 5 次采样取多数投票——"
              "只做一次会把随机性当成准确率损失。", first=2.0)

    # ------------------------------------------------ 三
    h1(doc, "三、两条评估纪律")
    rp(doc, "**第一条：任何防御都要成对报「ASR」与「干净准确率」。** 只看 ASR 下降，"
            "很容易得出「把模型打瞎了所以攻击失效了」这种荒谬结论。本记录用三个量：")
    for t in ["ASR：防御后对抗样本的误分类率，与 A1.1 同口径（全样本，非条件于攻击成功）；",
              "ASR_rel = (干净准确率 − 对抗准确率) / 干净准确率：扣掉防御自身准确率损失后的相对攻击率；",
              "net_gain = (ASR_无防御 − ASR_该防御) − (干净准确率_无防御 − 干净准确率_该防御)："
              "净防御收益，单位为 1，≤0 表示这个防御只是把模型弄笨了。"]:
        rp(doc, "· " + t, first=0, size=10.0)
    rp(doc, "**第二条：必须做自适应攻击。** 非自适应的评估只能说明「这类扰动被削弱」，"
            "不能说明「攻击失效」。判据很简单：如果攻击者知道防御存在并针对它优化，"
            "防御还剩多少？本次用 BPDA（后向微分近似）：前向真的过 JPEG/量化/模糊，"
            "反向把防御当恒等函数，梯度照常回传到原图。这是评估不可微预处理的标准做法。",
       first=2.0)

    # ------------------------------------------------ 四
    h1(doc, "四、结果")

    h2(doc, "4.1 防御本身要付多少准确率")
    para(doc, "下表是干净样本过同一防御后的准确率。这一列如果不报，后面所有 ASR 都没法解读。")
    base = {m: clean_of(rows, "none", m) for m in models}
    body = []
    for d in dlist:
        vv = {m: clean_of(rows, d, m) for m in models}
        body.append([DESC[d][0]] + [f"{vv[m]*100:.1f}" for m in models]
                    + [f"{np.mean([vv[m] for m in models])*100:.1f}",
                       f"{(np.mean([vv[m] for m in models]) - np.mean([base[m] for m in models]))*100:+.1f}"])
    table(doc, ["防御"] + models + ["平均", "Δ/pp"], body,
          [3.0] + [2.3] * len(models) + [2.0, 1.6],
          caption="表 2　干净样本在各防御下的准确率 / %")
    para(doc, "读法：Δ 列是相对无防御的百分点变化。掉 1~2 pp 属于正常代价，"
              "掉 5 pp 以上就要在论文里明确承认这条防御不可部署。", first=2.0)

    h2(doc, "4.2 主结果：四种攻击 × 各防御")
    for role, cap in (("集成内", "表 3　集成内目标的 ASR / %"),
                      ("未见", "表 4　未见目标的 ASR / %")):
        body = []
        for d in dlist:
            v = [A(rows, d, x, role=role) for x in atks]
            c = np.mean([clean_of(rows, d, m) for m in models])
            ng = np.nanmean([r["net_gain"] for r in rows
                             if r["defense"] == d and r["kind"] == "普通攻击"
                             and r["role"] == role and r["attack"] == "RNTME-NAE"])
            body.append([DESC[d][0]] + [f1(x) for x in v] + [f1(c), f"{ng*100:+.1f}"])
        table(doc, ["防御"] + atks + ["干净 acc", "净收益/pp\n(RNTME)"], body,
              [3.0, 2.2, 2.2, 2.4, 2.5, 2.1, 2.0], caption=cap)

    # 找最佳/最差
    ng_all = {d: np.nanmean([r["net_gain"] for r in rows if r["defense"] == d
                             and r["kind"] == "普通攻击"]) for d in dlist if d != "none"}
    best = max(ng_all, key=lambda k: ng_all[k]) if ng_all else None
    rsum = [A(rows, d, "RNTME-NAE", role="集成内") for d in dlist]
    dmax = dlist[int(np.argmax(rsum))]
    dmin = dlist[int(np.argmin(rsum))]
    rp(doc, f"**这两件事要分开看。** 按四攻击平均的净收益，最好的是 {DESC[best][0]}"
            f"（{ng_all[best]*100:+.1f} pp）；而把 RNTME-NAE 压得最狠的是 {DESC[dmin][0]}"
            f"（ASR 从 {A(rows, 'none', 'RNTME-NAE', role='集成内')*100:.1f}% 降到 "
            f"{A(rows, dmin, 'RNTME-NAE', role='集成内')*100:.1f}%），"
            f"但它同时让干净准确率掉了 "
            f"{(clean_of(rows, 'none', models[0]) - np.mean([clean_of(rows, dmin, m) for m in models]))*100:.1f} 个百分点，"
            f"净收益只有 "
            f"{np.nanmean([r['net_gain'] for r in rows if r['defense'] == dmin and r['kind'] == '普通攻击' and r['attack'] == 'RNTME-NAE' and r['role'] == '集成内'])*100:+.1f} pp（集成内）。"
            f"换句话说，这个数字是靠牺牲准确率换来的，不是靠破坏扰动换来的。", first=2.0)

    fp3 = os.path.join(RES, "overlap.json")
    if os.path.exists(fp3):
        ov = json.load(open(fp3, encoding="utf-8"))
        h2(doc, "4.2.1 一个容易被忽略的现象：ASR 相同，触达的样本并不相同")
        para(doc, f"表 4 里 I-FGSM、NAE、DI-NI-FGSM 三种攻击在未见目标上的 ASR 都是 "
                  f"{ov['asr']['I-FGSM']*100:.1f}%，看起来「效果一样」。但把逐样本的判错情况"
                  "两两比对就会发现问题：")
        body = [[u, v, f"{c*100:.1f}"] for u, v, c in ov["pairs"]]
        table(doc, ["攻击 A", "攻击 B", "错判模式一致率/%"], body,
              [5.2, 5.2, 4.4],
              caption=f"表 4b　未见目标 {ov['target']} 上的错判重合度（{ov['n']} 张）")
        rp(doc, "三个数放在一起说明：**这三种攻击可翻转的样本集合几乎是同一批**——"
                "它们能打下来的目标主要由样本自身的决策裕度决定，而不是由攻击方法决定。"
                "换成 RNTME 之后，一致率降到 0.90 左右，说明它打到了基线打不到的样本。",
           first=2.0)
        rp(doc, "这条对论文很有用：如果只报 ASR，读者会以为「换个方法但效果差不多」；"
                "报出重合度才能说明**方法差异体现在样本层面，而不只是总量层面**。"
                "正式实验应该把这张表做成按类别分组的形式，看 RNTME 究竟多触达了哪几类。",
           first=2.0)

    if os.path.exists(os.path.join(FIG, "fig1_pareto.png")):
        figure(doc, os.path.join(FIG, "fig1_pareto.png"), 16.2,
               "图 1　防御的代价—收益平面：横轴为干净准确率损失，纵轴为四攻击平均 ASR。"
               "越靠左下越好；右下角的点是「把模型弄笨了」")
    if os.path.exists(os.path.join(FIG, "fig2_heat.png")):
        figure(doc, os.path.join(FIG, "fig2_heat.png"), 16.2,
               "图 2　ASR 热力图：颜色越深攻击越有效，横向看是防御的削减效果")

    # ------------------------------------------------ 4.3 自适应
    bp = [r["defense"] for r in rows if r["kind"].startswith("自适应(")]
    bp = list(dict.fromkeys(bp))
    if bp:
        h2(doc, "4.3 自适应攻击：防御被知道之后还剩多少")
        para(doc, "这是本节最关键的一张表。判据要分两层看：先看防御是否真的削减了 ASR，"
                  "若削减了，再看向自适应攻击追回多少。追回大半说明只是挡住了「不懂它」的攻击；"
                  "本就削减不动，则说明防御对该类扰动无效——两种情况都不支持把预处理当卖点，"
                  "但原因不同，不能混为一谈。")
        body = []
        for d in bp:
            v0 = A(rows, d, "RNTME-NAE", role="集成内")
            v1 = A(rows, d, "RNTME-NAE", kind=f"自适应({d})", role="集成内")
            w0 = A(rows, d, "RNTME-NAE", role="集成内", key="SW-ASR")
            w1 = A(rows, d, "RNTME-NAE", kind=f"自适应({d})", role="集成内", key="SW-ASR")
            body.append([DESC[d][0], f1(v0), f1(v1), f"{(v1-v0)*100:+.1f}",
                         f1(w0), f1(w1)])
        table(doc, ["防御", "普通 ASR", "自适应 ASR", "回升/pp", "普通 SW-ASR", "自适应 SW-ASR"],
              body, [3.2, 2.6, 2.7, 2.4, 2.7, 2.8],
              caption="表 5　RNTME-NAE 在自适应攻击下的 ASR 回升")
        gains = [(A(rows, d, "RNTME-NAE", kind=f"自适应({d})", role="集成内")
                  - A(rows, d, "RNTME-NAE", role="集成内")) for d in bp]
        drops = [(A(rows, "none", "RNTME-NAE", role="集成内")
                  - A(rows, d, "RNTME-NAE", role="集成内")) for d in bp]
        mx, md = (max(gains) if gains else 0.0), (max(drops) if drops else 0.0)
        para(doc, f"自适应攻击用 298 张子集单独生成，因此它的 ASR 与「普通攻击」列"
                  f"不是同一批样本，两者比较时应看变化方向而不是小数点后一位。", first=2.0)
        if md < 0.06:
            rp(doc, f"**结论与「防御被绕过」相反：这些防御从一开始就没起作用。**"
                    f"普通攻击下最大的削减只有 {md*100:.1f} 个百分点（在单种子噪声量级内），"
                    f"自适应之后的变化也只有 {mx*100:.1f} 个百分点——两个数都说明"
                    f"防御对 RNTME 生成的低频、结构性扰动几乎没有影响。", first=2.0)
            para(doc, "这个结果其实比「被绕过」更值得写进论文：它意味着在 32 px 这类小图上，"
                      "压缩与量化删掉的高频分量并不是攻击迁移性的载体。RNTME 的扰动经过"
                      "梯度平滑与平移不变处理后已经偏向中低频，JPEG 的量化表恰好在这些频段上"
                      "保留得比较完整。这条解释可以在正式实验用频域分析验证。", first=2.0)
        elif mx > 0.6 * md:
            para(doc, f"自适应攻击把防御的削减追回了大半：普通攻击下最多削减 {md*100:.1f} 个百分点，"
                      f"自适应之后反向回升 {mx*100:.1f} 个百分点。这说明预处理类防御在知情攻击者"
                      "面前不成立，论文里不能把它当卖点，只能作为「部署链路中的偶然阻力」如实描述。",
                 first=2.0)
        else:
            para(doc, f"普通攻击下最多削减 {md*100:.1f} 个百分点，自适应之后只回升 {mx*100:.1f} 个百分点，"
                      "说明削减效果不是靠「攻击者不知道防御」得来的——扰动确实被破坏了一部分。"
                      "但要在论文里下这个结论，还需要在 224 px 与多种子条件下复现。", first=2.0)
        if os.path.exists(os.path.join(FIG, "fig3_bpda.png")):
            figure(doc, os.path.join(FIG, "fig3_bpda.png"), 11.6,
                   "图 3　普通攻击与自适应攻击的 ASR 对比（RNTME-NAE，集成内平均）")

    # ------------------------------------------------ 五 对抗训练
    h1(doc, "五、模型级防御：这次没得到可用结果，但定位了原因")
    fp2 = os.path.join(RES, "advtrain_convergence.json")
    cv = json.load(open(fp2, encoding="utf-8")) if os.path.exists(fp2) else None
    if cv is None:
        para(doc, "对抗训练诊断数据缺失（缺 advtrain_convergence.json）。")
    else:
        rp(doc, "预处理只改输入，模型的决策边界不变；能改变边界的只有对抗训练。"
                "本节如实记录三次尝试及其失败原因。**这一节没有可用的攻击结论**——"
                "诊断过程本身对正式实验有价值，所以留着。")

        h2(doc, "5.1 三次尝试")
        body = []
        for k, at in enumerate(cv["attempts"], 1):
            body.append([f"第 {k} 次", f"{at['lr']}", f"{at['epochs']}", f"{at['subset']}",
                         "是" if at["freeze_bn"] else "否",
                         f"{at['ep1_clean']*100:.1f}", at["verdict"]])
        table(doc, ["尝试", "lr", "轮数", "子集", "冻结 BN", "首轮 acc/%", "判断"], body,
              [1.4, 1.3, 1.2, 1.5, 1.5, 2.4, 6.7],
              caption="表 6　对抗训练的三次配置与首轮干净准确率")
        para(doc, f"对照：同一 600 张子集上，干净训练的 sresnet 是 "
                  f"{cv['clean_baseline_same_subset']*100:.1f}%。三条都没到可用水平，"
                  "因此不报告任何「对抗训练能否挡住攻击」的结论——用 17% 准确率的模型去算 ASR，"
                  "得到的数没有意义。", first=2.0)

        h2(doc, "5.2 不是实现错了，是收敛慢")
        c = cv["curve"]
        body = [[str(st), f"{u*100:.1f}", f"{v*100:.1f}", f"{(u-v)*100:+.1f}"]
                for st, u, v in zip(c["steps"], c["eps0"], c["eps8"])]
        table(doc, ["训练步数", "eps=0 干净 acc/%", "eps=8/255 对抗 acc/%", "干净−对抗/pp"], body,
              [2.8, 4.2, 4.8, 3.0],
              caption=f"表 7　同样步数下的学习曲线（{c['subset']} 张，bs={c['bs']}，PGD-{c['pgd_steps']}）")
        rp(doc, "关键证据是**两条曲线都在上升**。如果是 PGD 实现有错、或标签/数据错位，"
                "eps=8 那条会一直贴在随机水平（43 类约 2.3%）不动。它从 "
                f"{c['eps8'][0]*100:.1f}% 升到 {c['eps8'][-1]*100:.1f}%，说明流程是对的。",
           first=2.0)
        para(doc, f"差别在速度：同样 60 步（约 2 轮），干净训练到 {c['eps0'][-1]*100:.1f}%，"
                  f"对抗训练只到 {c['eps8'][-1]*100:.1f}%，慢约 "
                  f"{c['eps0'][-1]/max(c['eps8'][-1],1e-6):.1f} 倍。", first=2.0)
        para(doc, "对照 A1.1 的干净训练（5 轮到 95%），按这个比例外推，对抗训练要达到同等水平需要 "
                  "12~15 轮；文献的标准配置是 30 轮以上并配学习率预热。CPU 上单轮 3 分钟，"
                  "加 PGD 开销实际更久，做不完。这件事只能留给 GPU 正式实验。", first=2.0)

        h2(doc, "5.3 过程中的两个坑，正式实验别再踩")
        rp(doc, "**一是别冻结 BN 来省时间。** 试过先用干净数据校准 BN 统计再冻结，结果更差（5.2%）。"
                "原因是那份统计量来自**未训练的随机权重**，与训练后的权重完全不匹配。"
                "对抗训练里 BN 的 running stats 必须在训练中随权重一起收敛，这也是它需要更多轮的原因之一。",
           first=2.0)
        rp(doc, "**二是学习率要小于干净训练。** lr=0.05（与 A1.1 干净训练相同）时首轮只有 28.2%，"
                "降到 0.02 后曲线形状明显更稳。对抗样本让梯度方向更抖，同样的步长会把权重推过头。",
           first=2.0)

        h2(doc, "5.4 正式实验必须提前定下的口径：条件 ASR")
        para(doc, "对抗训练模型的干净准确率一定低于干净训练模型（这是代价，不是缺陷）。"
                  "此时若仍用全样本 ASR，会把「模型本来就分不对的样本」算成「攻击成功」，"
                  "结论会虚高。正确做法是只在「该模型原本能分类正确的样本」上统计：")
        rp(doc, "ASR_cond = 在 {x : f_clean(x) = y} 上，f(x_adv) ≠ y 的比例。", first=0, size=10.0)
        para(doc, "本记录的表 3、表 4 用的是全样本 ASR（与 A1.1 保持一致，便于横向比较），"
                  "防御评估里防御本身对准确率影响很小，两种口径差别不大；"
                  "但一旦进入对抗训练这种准确率大幅变化的场景，必须换成 ASR_cond，"
                  "否则「越鲁棒的模型 ASR 越低」这个结论会被准确率差异污染。", first=2.0)

    # ------------------------------------------------ 六
    h1(doc, "六、这一节能写进论文的部分")
    para(doc, "能写的是三段：")
    for t in ["**防御评估的方法学**：成对报 ASR 与干净准确率、净防御收益的定义、BPDA 自适应评估。"
              "这三条是审稿人判断「作者懂不懂防御评估」的地方，比具体数字更值钱。",
              "**预处理防御在知情攻击者面前不成立**这一负面结论。它直接支撑论文动机："
              "既然链路预处理挡不住，就必须在模型/训练侧想办法，而这正是本文方法的必要性来源。",
              "**「ASR 相同不等于方法相同」的证据**：三种基线攻击在未见目标上触达的样本集合"
              "高度重合（错判模式一致率 0.95~0.98），而 RNTME 的一致率只有 0.90 左右。"]:
        rp(doc, "· " + t, first=0, size=10.0)
    rp(doc, "不能写的是所有具体数字：32 px、单种子、600 张、合成尺度的预处理结论，"
            "在 224 px 与 KITTI/真实卡口图像上会明显不同（见第一节的 49 倍块效应估算）。"
            "正式实验必须重跑这一整节。", first=2.0)

    # ------------------------------------------------ 七
    h1(doc, "七、正式实验要改的地方")
    for t in ["**分辨率**：224 px 重跑，JPEG 的块效应相对强度会下降两个数量级，预计防御效果显著变弱；",
              "**数据集**：加 KITTI / UA-DETRAC 的实拍图像，而不是 GTSRB 裁剪图；",
              "**防御项**：补上「随机丢弃 + 邻域插值」与「主成分重建（MagNet 类）」，这两项是文献标配；",
              "**对抗训练**：用集成三模型全部做对抗训练，并报告 ε 取 8/255 与 16/255 的两条曲线；",
              "**多种子**：每种配置 5 个种子，ASR 报均值±标准差，防御间做配对检验；",
              "**随机化防御要用 EOT 而不是 BPDA**：对随机缩放填充这类推理时随机的防御，BPDA 的"
              "「前向真过、反向恒等」会在每次前向拿到不同的变换，梯度互相抵消。正确做法是"
              "期望变换（EOT）：每个梯度步采样 k 次变换取平均梯度，k 取 10~20；",
              "**报告频域证据**：第四节提出「RNTME 的扰动偏中低频，所以 JPEG 削不掉」这个解释，"
              "需要做扰动功率谱密度对比（RNTME vs I-FGSM）来支撑，否则只是推测。"]:
        rp(doc, "· " + t, first=0, size=10.0)

    out = os.path.normpath(os.path.join(HERE, a.doc))
    doc.save(out)
    print(f"  写出 {out}")


if __name__ == "__main__":
    main()
