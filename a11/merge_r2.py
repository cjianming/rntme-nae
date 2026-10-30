# -*- coding: utf-8 -*-
"""合并 r1（6 方法）与 r2（ADMIX / SGM 两个 2021 年后基线）的逐样本结果，
按论文 TABLE III 的口径重算全部数字，并做配对 Wilcoxon + Bonferroni 检验。

用法：python -m a11.merge_r2
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HERE_ROOT = os.path.dirname(HERE)

import sys
sys.path.insert(0, HERE_ROOT)
from a11 import swasr as S  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
R1 = ["I-FGSM", "MI-FGSM", "NAE", "DI-NI-FGSM", "RNTME-EQ", "RNTME-NAE"]
R2 = ["ADMIX", "SGM"]
ALL = R1 + R2
ENS = ["sresnet", "svgg"]
UNSEEN = "ssep"

W = np.array(S.build_matrix())


def load_preds() -> Dict:
    """{(attack, seed, target): pred array}"""
    out = {}
    for sd in SEEDS:
        z = np.load(os.path.join(RES, f"r1s{sd}_raw.npz"), allow_pickle=True)
        for k in z.files:
            if "|" in k and k.count("|") == 2:
                a, s, nm = k.split("|")
                if int(s) == sd:
                    out[(a, sd, nm)] = z[k]
    z2 = np.load(os.path.join(RES, "r2_raw.npz"), allow_pickle=True)
    for k in z2.files:
        if "|" in k and k.count("|") == 2:
            a, s, nm = k.split("|")
            out[(a, int(s), nm)] = z2[k]
    return out


def load_true_clean():
    z = np.load(os.path.join(RES, "r1s0_raw.npz"), allow_pickle=True)
    return z["true"], z[f"clean|{UNSEEN}"]


def load_times() -> Dict:
    t = {}
    for sd in SEEDS:
        with open(os.path.join(RES, f"r1s{sd}_time.json"), encoding="utf-8") as f:
            for k, v in json.load(f).items():
                a, s = k.split("|")
                t[(a, int(s))] = v
    with open(os.path.join(RES, "r2_time.json"), encoding="utf-8") as f:
        for k, v in json.load(f).items():
            a, s = k.split("|")
            t[(a, int(s))] = v
    return t


def main():
    preds = load_preds()
    Y, clean_unseen = load_true_clean()
    times = load_times()
    n = len(Y)
    ok = clean_unseen == Y

    rows = []
    for a in ALL:
        if a not in {k[0] for k in preds}:
            continue
        per_seed = {}
        for nm in ENS + [UNSEEN]:
            per_seed[nm] = {"asr": [], "cond": [], "sw": [], "dmr": []}
        for sd in SEEDS:
            for nm in ENS + [UNSEEN]:
                p = preds[(a, sd, nm)]
                w = W[Y, p]
                per_seed[nm]["asr"].append(float((p != Y).mean()))
                per_seed[nm]["cond"].append(float((p[ok] != Y[ok]).mean()) if ok.any() else float("nan"))
                per_seed[nm]["sw"].append(float(w.mean()))
                per_seed[nm]["dmr"].append(float((w >= 0.8).mean()))
        ens_mean = np.mean([np.mean(per_seed[nm]["asr"]) for nm in ENS])
        ms = np.mean([times[(a, sd)] for sd in SEEDS]) * 1000
        rows.append({
            "method": a,
            "ens": ens_mean * 100,
            "ens_std": np.std([np.mean([per_seed[nm]["asr"][i] for nm in ENS])
                               for i in range(len(SEEDS))], ddof=1) * 100,
            "unseen": np.mean(per_seed[UNSEEN]["asr"]) * 100,
            "unseen_std": np.std(per_seed[UNSEEN]["asr"], ddof=1) * 100,
            "cond": np.mean(per_seed[UNSEEN]["cond"]) * 100,
            "cond_std": np.std(per_seed[UNSEEN]["cond"], ddof=1) * 100,
            "sw": np.mean(per_seed[UNSEEN]["sw"]) * 100,
            "sw_std": np.std(per_seed[UNSEEN]["sw"], ddof=1) * 100,
            "dmr": np.mean(per_seed[UNSEEN]["dmr"]) * 100,
            "dmr_std": np.std(per_seed[UNSEEN]["dmr"], ddof=1) * 100,
            "ms": ms,
        })
        r = rows[-1]
        print(f"{a:<11s} ens {r['ens']:5.1f}±{r['ens_std']:.1f}  unseen {r['unseen']:5.1f}±{r['unseen_std']:.1f}"
              f"  cond {r['cond']:5.1f}±{r['cond_std']:.1f}  SW {r['sw']:5.1f}±{r['sw_std']:.1f}"
              f"  DMR {r['dmr']:4.1f}±{r['dmr_std']:.1f}  {r['ms']:6.0f}ms")

    # ---- 显著性：RNTME-NAE 对照其余方法，在未见目标上，指标 ASR 与 SW-ASR
    ref = "RNTME-NAE"
    others = [a for a in rows if a["method"] != ref]
    m = len(others) * 2
    print(f"\n== 配对 Wilcoxon（ref=RNTME-NAE, Bonferroni m={m}）==")
    res = []
    for o in others:
        a = o["method"]
        for tag, key in (("ASR", None), ("SW-ASR", None)):
            d1 = np.mean([(preds[(ref, sd, UNSEEN)] != Y).astype(float) if tag == "ASR"
                          else W[Y, preds[(ref, sd, UNSEEN)]] for sd in SEEDS], 0)
            d0 = np.mean([(preds[(a, sd, UNSEEN)] != Y).astype(float) if tag == "ASR"
                          else W[Y, preds[(a, sd, UNSEEN)]] for sd in SEEDS], 0)
            diff = d1 - d0
            if np.allclose(diff, 0):
                p = 1.0
            else:
                _, p = wilcoxon(d1, d0, zero_method="wilcox", alternative="two-sided",
                                correction=False, method="approx")
            res.append({"ref": ref, "other": a, "metric": tag,
                        "delta_pp": float(diff.mean() * 100),
                        "p_raw": float(p), "p_bonf": float(min(1.0, p * m)),
                        "signif": bool(p * m < 0.05)})
            print(f"  {ref} vs {a:<11s} {tag:<7s} Δ={diff.mean()*100:+6.2f}pp  "
                  f"p={p:.3g}  p_bonf={min(1.0,p*m):.3g}  {'显著' if p*m<0.05 else '不显著'}")

    with open(os.path.join(RES, "merged_table3.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "tests": res, "n": n}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RES, "merged_table3.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n{os.path.join(RES, 'merged_table3.csv')}")


if __name__ == "__main__":
    main()
