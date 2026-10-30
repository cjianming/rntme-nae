# -*- coding: utf-8 -*-
"""诊断 RNTME-NAE 各支路对源模型白盒成功率的贡献（32 px 试点口径）。"""
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A      # noqa: E402
from a11 import data as D         # noqa: E402
from a11 import run_seeds as RS   # noqa: E402

SIZE = int(os.environ.get("SIZE", 32))
N = int(os.environ.get("NSAMP", 24))
sigma_def = 1.5 if SIZE <= 32 else (3.0 if SIZE >= 64 else 2.0)

items = RS.build_subset(N, SIZE, seed=2)
ld = DataLoader(D.GTSRB(items, SIZE, False, cache=True), batch_size=N, shuffle=False)
X, Y = next(iter(ld))
m, acc, _ = RS.load("sresnet", SIZE)
with torch.no_grad():
    print("size=%d n=%d clean=%.3f" % (SIZE, len(Y), (m(X).argmax(1) == Y).float().mean()))

base = dict(eps=16 / 255, alpha=2 / 255, T=15, mu=0.9, B=5, tau=2.0, beta=0.7,
            sigma=sigma_def, early_stop=False, seed=0)


def asr(xa):
    with torch.no_grad():
        return (m(xa).argmax(1) != Y).float().mean().item()


for tag, over in [("I-FGSM", None), ("NAE", None),
                  ("RNTME 默认 sigma=%.1f" % sigma_def, {}),
                  ("RNTME sigma=0", dict(sigma=0.0)),
                  ("RNTME B=1", dict(B=1)),
                  ("RNTME T=10", dict(T=10))]:
    cfg = dict(base)
    if over is not None:
        cfg.update(over)
    fn = {"I-FGSM": A.ifgsm, "NAE": A.nae}.get(tag, A.rntme_nae)
    xa = fn({"sresnet": m}, X, Y, cfg)
    print("%-24s ASR=%.3f  L2=%.2f" % (tag, asr(xa),
                                       (xa - X).flatten(1).norm(dim=1).mean().item()))
