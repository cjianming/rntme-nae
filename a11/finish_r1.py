# -*- coding: utf-8 -*-
"""R1 收尾一键流程：合并 → 回填 → 重建 docx → 重新生成答复信。

只在本机跑；任一步失败即停。
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def sh(cmd, cwd=ROOT, env=None):
    e = os.environ.copy()
    e.update(env or {})
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=e, text=True, encoding="utf-8",
                       errors="replace", capture_output=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit("失败：%s" % " ".join(cmd))
    return out


def main():
    n = sys.argv[1] if len(sys.argv) > 1 else "150"

    sh([PY, "-m", "a11.merge_seeds", "--inputs", "r1s0,r1s1,r1s2,r1s3,r1s4",
        "--out", "r1", "--heldout", "sinc"])
    sh([PY, "a11/fill_r1.py"])

    # 正文样本量同步为实际值
    p = os.path.join(ROOT, "paper.md")
    s = open(p, encoding="utf-8").read()
    s = s.replace("按分层抽样取 300 幅测试图像", "按分层抽样取 %s 幅测试图像" % n)
    s = s.replace("64 px，300 幅，5 种子", "64 px，%s 幅，5 种子" % n)
    s = s.replace("4 个架构互异的分类器、300 幅测试图像", "4 个架构互异的分类器、%s 幅测试图像" % n)
    open(p, "w", encoding="utf-8", newline="").write(s)

    sh(["pandoc", "--from", "markdown-smart", "paper.md", "-o", "build/paper_raw.docx"])
    sh([PY, "postprocess.py"], env={"PAPER_DST": "基于NAE改进的车联网对抗样本生成方法_信息安全学报投稿版.docx"})
    sh([PY, "check.py"])
    sh([PY, "make_response.py"])
    left = open(p, encoding="utf-8").read().count("{{")
    print("\n残余占位符：", left)
    print("完成")


if __name__ == "__main__":
    main()
