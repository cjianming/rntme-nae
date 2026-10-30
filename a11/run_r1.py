# -*- coding: utf-8 -*-
"""R1 实验并行驱动：5 个种子各起一个进程，线程数降到 2，避免相互抢核。

用法：
    python a11/run_r1.py --size 64 --n 200
    python a11/run_r1.py --scan            # 只跑迭代预算扫描
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def launch(args, extra_env=None):
    env = os.environ.copy()
    env.update(extra_env or {})
    return subprocess.Popen([PY, "-m", "a11.run_seeds"] + args, cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--scan-n", type=int, default=60)
    ap.add_argument("--scan-attacks", default="RNTME-NAE")
    ap.add_argument("--tscan", default="5,8,11,15")
    args = ap.parse_args()

    env = {"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
           "TORCH_NUM_THREADS": str(args.threads)}

    if args.scan:
        cmd = ["--mode", "scan", "--size", str(args.size), "--n", str(args.scan_n),
               "--tscan", args.tscan, "--attacks", args.scan_attacks, "--out", "r1"]
        print("扫描：", " ".join(cmd), flush=True)
        p = launch(cmd, env)
        for line in p.stdout:
            print(line.rstrip(), flush=True)
        return p.wait()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    procs = {}
    t0 = time.time()
    for sd in seeds:
        cmd = ["--mode", "seeds", "--size", str(args.size), "--n", str(args.n),
               "--seeds", sd, "--bs", str(args.bs), "--out", "r1s%s" % sd]
        procs[sd] = launch(cmd, env)
        print("启动 seed=%s pid=%d" % (sd, procs[sd].pid), flush=True)

    rc = {}
    for sd, p in procs.items():
        out, _ = p.communicate()
        rc[sd] = p.returncode
        log = os.path.join(HERE, "r1s%s.log" % sd)
        open(log, "w", encoding="utf-8").write(out or "")
        print("seed=%s 结束 rc=%d  %s" % (sd, p.returncode, log), flush=True)
    print("总耗时 %.1f min，返回码 %s" % ((time.time() - t0) / 60, rc))
    return 0 if all(v == 0 for v in rc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
