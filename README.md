<div align="center">

# RNTME-NAE

**A task-level evaluation protocol for black-box attacks on vehicular perception — and its negative result.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877774.svg)](https://doi.org/10.5281/zenodo.23877774)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CPU only](https://img.shields.io/badge/hardware-CPU--only-lightgrey.svg)](#environment)
[![Code style: type hints](https://img.shields.io/badge/code-typed-orange.svg)](https://docs.python.org/3/library/typing.html)

</div>

Reference implementation for

> **Dynamic Ensemble Gradient Attacks for Vehicular Perception: A Task-Level
> Evaluation Protocol and a Bounded Negative Result**
> Jianming Cui, Ming Liu — submitted to *IEEE Transactions on Intelligent Transportation Systems*

---

## The result this repository exists to support

An ensemble attack transfers success **within its own surrogate batch**, not across
architectural boundaries. Measured on GTSRB at 64 px with 150 images and five seeds, on a
target that takes no part in generation, the eight compared methods land in
**40.0 % – 44.9 %** — the proposed one at 44.8 % — and no comparison survives Bonferroni
correction. A single-model baseline from 2021 (Admix) comes within 1.5 points at roughly a
fifth of the wall-clock cost.

We publish this because the negative result is the contribution. The protocol, the
consequence-weighted metrics and every per-sample record needed to check it are here.

## Repository layout

| Path | Paper section | Reproduces |
|---|---|---|
| `a11/` | §VI-F, §VI-G | GTSRB transfer benchmark (**Table III**), semantic weight tiers (**Table IV**). Contains `chk/` with three 64 px classifiers and `results/*_raw.npz` per-sample predictions |
| `a12/` | §VI-G | Object-detection extension (**Table V**): synthetic traffic scenes, two single-stage surrogates, one two-stage held-out detector |
| `a13/` | §VI-H | Seven input-level defences under BPDA adaptive attack |

Each module keeps its own detailed Chinese `README.md` documenting design choices and
ablations that did not fit in the paper. Start there to understand *why* a number is what
it is; come back here for how to run everything.

## Environment

```
Python 3.14    torch 2.13    numpy 2.x    scipy 1.x
```

Everything runs on **CPU**; no GPU is required or assumed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torchvision` is only needed by `a12/`. `matplotlib`, `pillow` and `python-docx` are only
needed by the `make_figures.py` / `make_report.py` helpers.

## Data

No dataset is redistributed here.

**GTSRB** (`a11/`) — licence forbids redistribution. Download
`GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip` and place both under
`data/gtsrb/`; `a11/data.py` extracts and indexes them automatically.

**PASCAL VOC 2007** (`a12/`) — downloaded automatically by `torchvision` on first use.

## Checkpoints

| Module | Bundled? | How to obtain |
|---|---|---|
| `a11/chk/` | **Yes** — 5 checkpoints, ~11 MB | Ready to run |
| `a12/chk/` | **No** | Train locally, see below |
| `a13/` | n/a | Reuses `a11/chk/` |

`a12/chk/` is deliberately empty: the Faster R-CNN checkpoint alone is 73 MB, above the size
GitHub handles gracefully. Train the three detectors yourself first:

```bash
python -m a12.train --arch ssdlite    --n 400 --epochs 5 --seed 0 --out a12/chk/ssdlite_a.pt
python -m a12.train --arch ssdlite    --n 400 --epochs 5 --seed 1 --out a12/chk/ssdlite_b.pt
python -m a12.train --arch frcnn_mnv3 --n 400 --epochs 5 --seed 0 --out a12/chk/frcnn_mnv3.pt
```

Consequence of this choice: **Table V is reproducible but not pre-computed.** The archived
`a12/results/` npz files were produced from exactly these training commands; freshly
trained weights reproduce them up to training-noise variation.

## Quickstart

The three commands below marked **no data needed** run out of the box and are the fastest
way to check that the archive is self-consistent.

```bash
# Verify the metrics themselves (no data, no model needed)
python -m a11.swasr

# Table III — regenerate the eight-method statistics and their Bonferroni-corrected
# tests from the archived per-sample records  [no data needed]
python -m a11.merge_r2

# Recompute SW-ASR / DMR under the v2 matrix without re-running any attack  [no data needed]
python -m a11.recompute_v2 --prefix pilot
```

`python -m a11.merge_r2` regenerates `a11/results/merged_table3.csv` from
`r1s{0..4}_raw.npz` plus `r2_raw.npz`. We re-ran it against the archived copy: the output is
**byte-identical**, and it reproduces the reported verdict — eight methods inside 40.0–44.9 %,
every comparison non-significant after correction.

Re-running the attacks themselves needs GTSRB (see [Data](#data)):

```bash
# Table III / IV — GTSRB protocol, five seeds, 150 images, 64 px  [needs GTSRB]
python -m a11.run_seeds --size 64 --n 150 --T 15 --seeds 0,1,2,3,4 \
       --attacks "I-FGSM,MI-FGSM,NAE,DI-NI-FGSM,RNTME-EQ,RNTME-NAE,ADMIX,SGM" \
       --ensemble sresnet,svgg --heldout ssep --out r1

# Iteration-budget scan behind the saturation analysis of §VI-C  [needs GTSRB]
python -m a11.run_seeds --mode scan --n 120 --tscan 5,8,11,15 --out tscan

# Table V — detection  [needs the checkpoints trained above]
python -m a12.run --data synth --n 80 --T 10 --B 3 \
       --ens ssdlite_a,ssdlite_b --heldout frcnn_mnv3 \
       --ckpt ssdlite_a=a12/chk/ssdlite_a.pt,ssdlite_b=a12/chk/ssdlite_b.pt,frcnn_mnv3=a12/chk/frcnn_mnv3.pt \
       --tag pilot

# §VI-H — seven defences and their BPDA adaptive counterparts  [no external data]
python -m a13.run
```

## Random sources

Every stochastic step carries an explicit seed; nothing depends on implicit global RNG state.

| Stage | Seed | Where |
|---|---|---|
| Training subset of the GTSRB train split | 1 | `a11/train.py` |
| Train / validation split (10 %) | 0 | `a11/train.py` |
| Test subset (150 images) of the GTSRB test split | 2 | `a11/run_seeds.py` |
| Attack randomness, five repeats | 0, 1, 2, 3, 4 | `a11/run_seeds.py` |
| Admix image pool (300 images, from the **train** split, disjoint from the evaluation subset) | 7 | `a11/run_seeds.py` |
| Admix within-attack sampling | attack seed + 977 | `a11/attacks.py` |
| Detection: scene synthesis | 0 (+7777 for the test split) | `a12/run.py` |
| Defence: evaluation subset | 2 | `a13/run.py` |

All five attack seeds share the same 150 evaluation images, so reported ± values reflect
attack randomness only and **do not** include sampling variance.

## Semantics: why ASR is not enough

`a11/swasr_weights.csv` holds the full 43 × 43 confusion-weight matrix `W[true][predicted]`
behind both consequence-weighted metrics:

- **SW-ASR** = mean of `W[t][p]` — how much harm the errors carry
- **DMR** = fraction of samples with `W[t][p] ≥ 0.8` — the dangerous tail

Weights follow **traffic consequence, not class distance**. Reading a 30 km/h limit as 80 is
relaxation (ω = 1.0); reading 80 as 30 costs efficiency but not lives (ω = 0.3). The
self-test in `swasr.py` shows two batches with identical ASR of 1.00 whose SW-ASR differ
fivefold (0.30 vs 1.00) — the difference the plain success rate is blind to.

The matrix is versioned **v2**. v1 scored safety-side errors (any sign read as stop or as a
generic warning) at zero, which understates phantom braking; archived v1 records are kept in
`a11/results/pilot_v1.csv` so the change can be audited. Replace the matrix with an
independently scored one through `swasr.load_csv()` — the paper asks for at least three
independent raters with Kendall's *W* before any operational use.

## Scope: what cannot be reproduced here

The main tables (§VI-A–§VI-D, **Tables I–II**) are **not** reproducible from this
repository. They were measured on a self-collected dataset annotated from surveillance and
dashcam footage, which cannot be redistributed — every licence plate and face was removed,
and the imagery stays with the authors. What is published instead is the configuration
(architecture list, ε, step size, iteration budget, early-stopping window) in §V and §VI-A,
enough to rebuild an equivalent protocol on another dataset.

Further declared limitations, all stated in the paper rather than hidden here: no vision
transformer among the targets; the two "unseen" targets remain convnets related to the
source; the §VI-F benchmark runs at 64 px, below deployment resolution; and the
dynamic-vs-equal weighting comparison exists only at *M* = 2. See §VII-B for the full list.

## Intended use and responsible disclosure

This is an **offensive** evaluation toolkit for defensive research: it exists so that a
perception stack can be audited before deployment, and so that claims of transferability can
be checked against a protocol that keeps its targets honest.

Please note the deliberate restraints, inherited from the accompanying paper:

- Attacks are generated **offline**; no component issues online queries against a live service.
- **No vehicular imagery is distributed**, and the self-collected dataset is excluded entirely.
- Only the research code and public-dataset artefacts are public; the deployment wrapper is withheld.
- Report every defence alongside its clean accuracy. An ASR drop alone is not a defence — see
  `a13/README.md` on why `ASR_rel` and `net_gain` exist.

## License and citation

Code is **MIT** (see [LICENSE](LICENSE)). If it helps your work, cite the paper — machine-readable
metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@article{cui2026dynamic,
  title   = {Dynamic Ensemble Gradient Attacks for Vehicular Perception:
             A Task-Level Evaluation Protocol and a Bounded Negative Result},
  author  = {Cui, Jianming and Liu, Ming},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under review}
}
```

### Archiving and DOI

For a citable snapshot, the release is archived on Zenodo, which issues a DOI:

```bibtex
@software{cui2026rntme,
  title  = {RNTME-NAE: reference implementation of a task-level evaluation
            protocol for black-box attacks on vehicular perception},
  author = {Cui, Jianming and Liu, Ming},
  year   = {2026},
  doi    = {10.5281/zenodo.23877774},  % version DOI: freezes v1.0.0
  license= {MIT}
}
```

Zenodo issues two identifiers and **a paper must cite the version DOI**, not the concept DOI:

| DOI | Points to | Cite it |
|:---|:---|:---|
| Version DOI | one frozen release (`v1.0.0`) | in the paper — pins the exact code that produced the results |
| Concept DOI | always the newest version | in a README or slide — for "the software" generally |

To mint one, log in to [zenodo.org](https://zenodo.org) with GitHub, enable this repository under
Zenodo's GitHub settings, then publish a GitHub Release. Zenodo persists a snapshot and assigns the DOI
within minutes. The repository must be public, and only these four things may change afterwards: a new
release gets its own DOI, so anything already published stays reproducible.

Archiving can also be done without GitHub, by uploading an archive directly — useful for GitLab-hosted
or air-gapped code:

```bash
git tag -a v1.0.0 -m "First archived release"
git archive --format=zip --prefix=rntme-nae-v1.0.0/ -o zenodo_upload_v1.0.0.zip v1.0.0
```

> Cite the paper with the **version DOI** [10.5281/zenodo.23877774](https://doi.org/10.5281/zenodo.23877774),
> which freezes `v1.0.0`, the exact code that produced the published results. The concept DOI
> [10.5281/zenodo.23877773](https://doi.org/10.5281/zenodo.23877773) always points to the newest
> version and is meant for a README or slide, not for a citation.
