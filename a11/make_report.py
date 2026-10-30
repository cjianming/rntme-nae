# -*- coding: utf-8 -*-
"""把试点结果整理成一份可给导师/审稿人看的记录文档。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import swasr as S                 # noqa: E402
from a11.analyze import CN                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

SONG, HEI, LATIN = "宋体", "黑体", "Times New Roman"


def newdoc():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = sec.bottom_margin = Cm(2.4)
    st = doc.styles["Normal"]
    st.font.name = LATIN
    st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    st.paragraph_format.line_spacing = 1.45
    st.paragraph_format.space_after = Pt(2)
    return doc


def _f(run, east=SONG, latin=LATIN):
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)


def para(doc, text="", size=10.5, bold=False, east=SONG, align="just",
         first=2.0, before=0, after=2, line=1.45, color=None, indent=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Cm(first)
    pf.left_indent = Cm(indent)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY,
                    "left": WD_ALIGN_PARAGRAPH.LEFT,
                    "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    if text:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        _f(r, east)
        if color:
            r.font.color.rgb = color
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.2
    pf.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(13)
    r.bold = True
    _f(r, HEI)


def h2(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.2
    pf.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.bold = True
    _f(r, HEI)


def set_cell(cell, text, size=8.6, bold=False, align="left"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1.2)
    p.paragraph_format.space_after = Pt(1.2)
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.first_line_indent = Cm(0)
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    _f(r)
    tcPr = cell._tc.get_or_add_tcPr()
    va = OxmlElement("w:vAlign")
    va.set(qn("w:val"), "center")
    tcPr.append(va)


def table(doc, header, rows, widths, size=8.6, caption=None, first_center=True,
          keep=False):
    if caption:
        p = para(doc, caption, size=9, bold=True, east=HEI, align="center",
                 first=0, before=7, after=2.5, line=1.2)
        p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    bd = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "8C8C8C")
        bd.append(e)
    t._tbl.tblPr.append(bd)
    for i, h in enumerate(header):
        set_cell(t.rows[0].cells[i], h, size=size, bold=True, align="center")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            al = "center" if (first_center and i == 0) else (
                "center" if str(v).replace(".", "").replace("%", "").replace("−", "").replace("-", "").isdigit() else "left")
            set_cell(cells[i], str(v), size=size, align=al)
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(int(sum(widths) * 567)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * 567)))
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    if keep:
        for row in t.rows[:-1]:
            for c in row.cells:
                for pp in c.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(doc, path, width_cm, caption):
    if not os.path.exists(path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Cm(0)
    p.add_run().add_picture(path, width=Cm(width_cm))
    cp = para(doc, caption, size=9, east=HEI, align="center", first=0,
              before=0, after=8, line=1.2)


def direction_stats(prefix):
    """统计每种攻击的误判方向：限速类误判中「放宽」与「收紧」各多少。"""
    W = S.build_matrix()
    SP, SV = S.SPEED_CLASSES, S.SPEED_VALUE
    out = {}
    for f in sorted(os.listdir(RES)):
        if not (f.startswith(prefix + "_") and f.endswith(".npz")):
            continue
        atk = f[len(prefix) + 1:].rsplit("_", 1)[0]
        d = np.load(os.path.join(RES, f))
        o = out.setdefault(atk, {"succ": 0, "wsum": 0.0, "danger": 0, "perm": 0, "strict": 0})
        for a, b in zip(d["true"], d["pred"]):
            a, b = int(a), int(b)
            if a == b:
                continue
            w = W[a][b]
            o["succ"] += 1
            o["wsum"] += w
            o["danger"] += int(w >= 0.8)
            if a in SP and b in SP:
                if SV[b] > SV[a]:
                    o["perm"] += 1
                elif SV[b] < SV[a]:
                    o["strict"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pilot")
    ap.add_argument("--out", default="A11_交通标志实验记录.docx")
    args = ap.parse_args()

    with open(os.path.join(RES, f"{args.prefix}_summary.json"), encoding="utf-8") as f:
        J = json.load(f)
    cfg, summ, rows = J["config"], J["summary"], J["rows"]
    clean = J["clean_acc"]

    doc = newdoc()
    para(doc, "A1.1 交通标志识别任务级攻击实验", size=18, bold=True, east=HEI,
         align="center", first=0, before=0, after=3, line=1.2)
    para(doc, "GTSRB · 试点运行记录（非论文数据）", size=11, east=HEI,
         align="center", first=0, before=0, after=6, line=1.3)
    para(doc, f"配置：{cfg['size']} px ｜ 集成 {cfg['ensemble']} ｜ 未见目标 {cfg['heldout']} ｜ "
              f"测试样本 {cfg['n']} ｜ T={cfg['T']} ｜ ε={cfg['eps']:.4f}",
         size=8.8, align="center", first=0, before=0, after=12,
         color=RGBColor(0x66, 0x66, 0x66))

    # ---------------- 1 做了什么 ----------------
    h1(doc, "一、这一轮做了什么")
    para(doc, "把《T-ITS 投稿补充方案》里的 A1.1 落成可运行的代码，并在本机跑通全链路："
              "训练替代模型 → 生成对抗样本 → 在集成内目标与未见目标上分别统计 → "
              "按交通语义加权算指标 → 提取混淆对。目的是先把「攻击做成什么、指标怎么算」这两件事定下来，"
              "而不是产出论文数字。")
    para(doc, "本机没有 NVIDIA 显卡（仅 Intel UHD 620），因此采用 32 px + 4 个架构互异的小网络 + "
              "12 000 幅训练子集的缩减配置。这个配置得出的绝对数值不能写进论文，"
              "但它足以验证代码正确性与指标行为，且与正式配置共用同一套脚本。")

    # ---------------- 2 指标 ----------------
    h1(doc, "二、指标：为什么 ASR 不够")
    para(doc, "ASR 只回答「有没有骗过」，不回答「骗得有多危险」。把限速 80 读成 30 是效率损失，"
              "把限速 30 读成 80 是违法与事故，两者在 ASR 里是同一个 1。因此按交通语义给混淆对赋权：")
    table(doc,
          ["语义档位", "权重", "典型混淆对", "交通后果"],
          [["严重放宽", "1.0", "限速 30 → 限速 80；停车让行 → 干道优先；禁止超车 → 解除禁止超车",
            "可直接导致超速、闯停、违规变道"],
           ["明显放宽", "0.8", "限速 30 → 限速 50；右转 → 左转", "需驾驶员纠正，否则走错或抢行"],
           ["同族混淆", "0.5", "行人 → 儿童；警告 → 另一类警告", "风险量级相当，仍保留警示"],
           ["更严格", "0.3", "限速 80 → 限速 30", "安全侧错误，仅效率损失"],
           ["误刹车", "0.2", "限速 80 → 停车；干道优先 → 停车让行", "不超速，但幽灵急停本身是追尾诱因"],
           ["无害提示", "0.1", "信号灯 → 一般危险警告", "只让车减速，几乎无危害"]],
          [2.2, 1.5, 8.4, 4.4], caption="表 1　SW-ASR 的六档语义权重（v2）")
    para(doc, "v1 曾把「任何标志 → 停车 / 一般警告」一律记 0，认为那是安全侧、不必惩罚。"
              "这条规则过宽：把限速 80 读成停车标志，车辆会在高速路段无故急停，本身就是追尾诱因；"
              "把信号灯读成「一般危险警告」也会造成不必要的制动。v2 起 0 只留给对角线（预测正确），"
              "安全侧错误统一落在 0.1～0.3 的低风险档。改动使 SW-ASR 绝对值整体上移约 0.01～0.02，"
              "四种攻击的相对次序与结论均未改变。", first=2.0)
    para(doc, "在此基础上定义两项指标：SW-ASR = 加权命中率；DMR = 权重 ≥ 0.8 的样本比例（只数危险那一档）。"
              "两者与 ASR 并列报告，不替代 ASR。")

    # ---------------- 3 结果 ----------------
    h1(doc, "三、试点结果")
    para(doc, "先看干净准确率，确认替代模型本身是有效的：")
    cl = "；".join(f"{k} {v*100:.1f}%" for k, v in clean.items())
    para(doc, cl + "。四个模型均在 43 类上训练，未见目标（" + cfg["heldout"] + "）"
              "从未参与对抗样本生成。", first=2.0)

    hdr = ["攻击", "集成内 ASR/%", "SW-ASR/%", "DMR/%", "未见目标 ASR/%",
           "未见 SW-ASR/%", "L₂", "SSIM", "ms/样本"]
    body = []
    for r in summ:
        body.append([r["attack"],
                     f"{r['in_ASR']*100:.1f}", f"{r['in_SW']*100:.1f}", f"{r['in_DMR']*100:.1f}",
                     f"{r['out_ASR']*100:.1f}", f"{r['out_SW']*100:.1f}",
                     f"{r['L2']:.2f}", f"{r['SSIM']:.3f}", f"{r['ms']:.0f}"])
    table(doc, hdr, body, [2.4, 1.9, 1.8, 1.5, 2.1, 2.1, 1.2, 1.2, 1.6],
          size=8.2, caption="表 2　四种攻击在 GTSRB 上的表现（试点配置）")

    # 关键对比一：集成内 vs 未见目标
    inb = [r for r in rows if r["role"] == "集成内"]
    outb = [r for r in rows if r["role"] == "未见"]
    gi = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    rn = [r for r in rows if r["attack"] == "RNTME-NAE"]
    base = [r for r in rows if r["attack"] != "RNTME-NAE"]
    best_in = max([r["in_ASR"] for r in summ if r["attack"] != "RNTME-NAE"])
    rn_in = [r["in_ASR"] for r in summ if r["attack"] == "RNTME-NAE"][0]
    rn_out = [r["out_ASR"] for r in summ if r["attack"] == "RNTME-NAE"][0]
    b_out = max([r["out_ASR"] for r in summ if r["attack"] != "RNTME-NAE"])

    h2(doc, "3.1 两个必须看的对比")
    para(doc, f"第一个对比是集成内与未见目标。RNTME-NAE 在集成内达到 {rn_in*100:.1f}%，"
              f"比最强基线（{best_in*100:.1f}%）高 {(rn_in-best_in)*100:.1f} 个百分点；"
              f"但在完全未参与生成的 {cfg['heldout']} 上，它只有 {rn_out*100:.1f}%，"
              f"而基线最高为 {b_out*100:.1f}%——优势归零，甚至略低。", first=2.0)
    para(doc, "这个结果不能忽略。它意味着 RNTME-NAE 的提升高度依赖「被优化过的模型」："
              "只要模型参与梯度加权，成功率就大幅上升；换成一个架构相似但从未见过的模型，收益消失。"
              "换句话说，在集成成员上测出来的数，衡量的是拟合能力，不是迁移能力。"
              "原稿的 85.0% 正是在集成成员上测的，这条试点结果与之前的文本审查指向同一个问题，"
              "只是这次是实验证据而不是推理。")
    para(doc, "必须说明的是，试点只有 1 个未见目标、1 个种子、32 px 输入，"
              "不能据此断言「全量下也没有优势」。但它把风险具体化了：正式实验必须把未见目标的结果"
              "与集成内结果并排报告，且不能只报后者。")

    # 关键对比二：误判方向
    ds = direction_stats(args.prefix)
    if ds:
        h2(doc, "3.2 误判方向的差异")
        para(doc, "第二个对比是「错成什么样」。只看 ASR，RNTME-NAE 是 63%，基线是 45～47%；"
                  "但把误判按方向拆开后，差异比数量上的差异更值得写：", first=2.0)
        body2 = []
        for atk in ["I-FGSM", "NAE", "DI-NI-FGSM", "RNTME-NAE"]:
            if atk not in ds:
                continue
            o = ds[atk]
            ratio = o["perm"] / max(o["strict"], 1)
            body2.append([atk, o["succ"], f"{o['wsum']/max(o['succ'],1):.3f}",
                          f"{o['danger']/max(o['succ'],1)*100:.1f}",
                          o["perm"], o["strict"], f"{ratio:.2f}"])
        table(doc, ["攻击", "成功数", "成功样本均权重", "危险占比/%",
                    "限速放宽", "限速收紧", "放宽/收紧"],
              body2, [2.9, 1.7, 2.6, 2.2, 1.9, 1.9, 2.0], size=8.2,
              caption="表 3　各攻击的误判方向（四个目标合并）")
        para(doc, "基线把限速标志改错时，绝大多数是「往低改」——限速 80 读成 50、限速 70 读成 30，"
                  "放宽/收紧比只有 0.55～0.58。这类错误车会开得更慢，属安全侧。"
                  "RNTME-NAE 的比值是 0.93，几乎一半的限速误判是往高改：限速 50 读成 80、"
                  "限速 70 读成 80。同样的「攻破」，后者才会让车超速。", first=2.0)
        para(doc, "需要克制的地方：成功样本的平均权重四者都在 0.50 上下，危险占比 21%～24%，"
                  "差别不大。真正的差别在「放宽方向的绝对次数」——RNTME-NAE 是 207 次，"
                  "基线是 122～128 次，约 1.7 倍。所以准确的表述是「RNTME-NAE 造成的放宽型误判"
                  "在数量上明显更多」，而不是「它每次误判都更危险」。")

    figure(doc, os.path.join(FIG, "figA_metrics.png"), 12.2,
           "图 1　同一攻击在 ASR、SW-ASR、DMR 三个口径下的对比")
    figure(doc, os.path.join(FIG, "figB_confusions.png"), 12.2,
           "图 2　RNTME-NAE 造成的主要混淆对及语义权重")

    # ---------------- 4 混淆对 ----------------
    conf_path = os.path.join(RES, f"{args.prefix}_confusions.json")
    if os.path.exists(conf_path):
        h1(doc, "四、攻击实际把什么改成了什么")
        para(doc, "这一节是交通读者最关心的部分：不是「成功率多少」，而是「限速 30 被改成了什么」。"
                  "下表把 RNTME-NAE 在各目标上最常见的混淆对合并统计。")
        with open(conf_path, encoding="utf-8") as f:
            C = json.load(f)
        W = S.build_matrix()
        merged = Counter()
        for tgt, rows2 in C.get("RNTME-NAE", {}).items():
            for r in rows2:
                merged[(r["true"], r["pred"])] += r["count"]
        body = []
        for (t, p), c in merged.most_common(10):
            w = W[t][p]
            # 与 v2 六档对应：1.0/0.8 放宽，0.5 同族，0.3 更严格，0.2 误刹车，0.1 仅减速
            tag = ("严重放宽" if w >= 0.8 else
                   "明显放宽/同族" if w >= 0.45 else
                   "更严格" if w >= 0.25 else
                   "误刹车" if w >= 0.15 else
                   "仅减速" if w > 0 else "安全侧")
            body.append([f"{CN[t]} → {CN[p]}", c, f"{w:.1f}", tag])
        table(doc, ["混淆对", "次数", "权重", "判定"], body, [6.6, 1.8, 1.8, 3.0],
              size=8.4, caption="表 4　RNTME-NAE 的主要混淆对（各目标合并）", keep=True)
        para(doc, "论文里这张表应当按类别归一化成「占该类被攻击样本的比例」，"
                  "并挑 3～4 个语义最重的案例配图（原图 / 对抗样本 / 判定标签）。", first=2.0)

    # ---------------- 5 与论文的关系 ----------------
    h1(doc, "五、这批数据能写进论文吗")
    para(doc, "不能。以下五项与正式实验不同，混用会被审稿人抓住：")
    for t in ["输入 32 px（高斯核 σ 相应取 1.5），正式实验为 224 px、σ=3；",
              "替代模型为 4 个自建小网，正式实验须为 ResNet50 / EfficientNet-B0 / VGG19 / "
              "MobileNetV2 / GoogLeNet 五模型集成；",
              "训练子集 12 000 幅、5 轮，正式实验为全量 39 209 幅、30 轮；",
              "集成 M=3、未见目标 1 个，正式实验 M=5、未见目标 2 个（ResNet101 与 DenseNet121）；",
              "单种子单次运行，正式实验需 5 个种子并报告均值±标准差与配对检验。"]:
        para(doc, "· " + t, first=0, indent=0.6)

    para(doc, "能写进论文的是三样东西：指标定义（SW-ASR / DMR）、权重表（需领域人员打分确认）、"
              "以及混淆对分析这个呈现方式。方法是新的，数字要换。", before=4)

    # ---------------- 6 下一步 ----------------
    h1(doc, "六、下一步")
    h2(doc, "6.1 在有显卡的机器上跑全量")
    para(doc, "把整个 a11/ 目录连同 data/gtsrb 的三个 zip 拷过去，两条命令即可：", first=2.0)
    para(doc, "python -m a11.train --arch full --size 224 --epochs 30 --bs 64", size=9.2,
         east=LATIN, align="left", first=0, indent=0.8, line=1.2)
    para(doc, "python -m a11.eval_attack --size 224 --n 5000 --T 15 --sigma 3 "
              "--ensemble resnet50,efficientnet_b0,vgg19,mobilenet_v2,googlenet "
              "--heldout resnet101,densenet121 --out full", size=9.2, east=LATIN,
         align="left", first=0, indent=0.8, line=1.2)
    h2(doc, "6.2 权重表要找人打分")
    para(doc, "a11/swasr_weights.csv 是规则推导的版本（当前 v2，已修正「安全侧记 0」的过宽规则）。"
              "请 3 名交通工程或交通安全方向的人独立改分，取均值并报告 Kendall's W。"
              "论文里必须写明「权重由谁定、一致性多少」，否则 SW-ASR 会被质疑成作者自定义的凑数指标。")
    h2(doc, "6.3 补三件试点没做的事")
    for t in ["ε 扫描（8/255、12/255、16/255），给出「隐蔽性—危害」曲线；",
              "视频序列上的时间一致性（UA-DETRAC），报告逐帧翻转率；",
              "与防御的交互（JPEG、位深缩减），验证低频扰动是否更耐压缩。"]:
        para(doc, "· " + t, first=0, indent=0.6)

    out = os.path.join(os.path.dirname(HERE), args.out)
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
