# -*- coding: utf-8 -*-
"""64 px 正式配置下的快速体检：M=3 集成，看源模型与未见目标上的 ASR 量级是否合理。"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a11 import attacks as A      # noqa: E402
from a11 import data as D         # noqa: E402
from a11 import run_seeds as RS   # noqa: E402

SIZE, N = 64, int(os.environ.get("NSAMP", 16))
ENS = os.environ.get("ENS", "sresnet,svgg,ssep").split(",")
HOLD = os.environ.get("HOLD", "sinc")
items = RS.build_subset(N, SIZE, seed=2)
ld = DataLoader(D.GTSRB(items, SIZE, False, cache=True), batch_size=16, shuffle=False)
X, Y = next(iter(ld))
models = {}
for nm in ENS + [HOLD]:
    models[nm], _, _ = RS.load(nm, SIZE)
with torch.no_grad():
    acc = {nm: float((models[nm](X).argmax(1) == Y).float().mean()) for nm in models}
print("size=%d n=%d clean=%s" % (SIZE, len(Y), {k: round(v, 3) for k, v in acc.items()}))

base = dict(eps=16 / 255, alpha=2 / 255, T=15, mu=0.9, B=5, tau=2.0, beta=0.7,
            sigma=3.0, early_stop=False, seed=0)
ens = {k: models[k] for k in ENS}
for a in ["I-FGSM", "MI-FGSM", "NAE", "DI-NI-FGSM", "RNTME-EQ", "RNTME-NAE"]:
    t0 = time.time()
    cfg = dict(base, source=ENS[0])
    if a == "RNTME-EQ":
        cfg["equal_weight"] = True
    xa = A.run(a, ens, X, Y, cfg)
    with torch.no_grad():
        pr = {nm: models[nm](xa).argmax(1) for nm in models}
    out = "  ".join("%s=%.2f" % (nm, float((pr[nm] != Y).float().mean())) for nm in models)
    print("%-11s %s  %.0f ms/样本" % (a, out, (time.time() - t0) / len(Y) * 1000))
