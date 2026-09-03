#!/usr/bin/env python3
"""
Timing sweep over bin/main: runs every requested (dataset, mode, version) combination, parses the
per iteration timings that main.c prints, and writes one tidy CSV row per iteration.

    python3 tools/benchmark.py                                   # defaults
    python3 tools/benchmark.py -d 500_10 150_10 -g 4 5 7         # pick datasets / GPU versions
    python3 tools/benchmark.py --cpu 0 1 2 3 --gpu 0 2 3 4 5 7 --hybrid 0 1
    python3 tools/benchmark.py -o outputs/timings.csv

CSV columns: dataset,mode,version,label,iter,time_s,align_len,identity
(iter = -1 is the warmup iteration; it is kept so it can be discarded explicitly.)

Then plot with:  python3 tools/plot_results.py
Run from the repository root.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

CPU_LABELS = {0: "sequential", 1: "simd", 2: "simd_parallel_dp", 3: "simd_parallel_node",
              4: "last_col", 5: "multi"}
GPU_LABELS = {0: "naive", 1: "naive(pinned)", 2: "parallel_node", 3: "parallel_async",
              4: "async_monolithic", 5: "async_batching", 7: "shared_mem",
              8: "last_col", 9: "warps", 10: "registers", 11: "persistent_kernels"}
HYBRID_LABELS = {0: "hybrid_base", 1: "hybrid_unified", 2: "hybrid_pinned",
                 3: "hybrid_advised"}
NOCOPY_LABELS = {0: "nc_naive", 1: "nc_naive", 2: "nc_level", 3: "nc_level", 4: "nc_level",
                 5: "nc_level", 6: "nc_shared_mem"}

ITER_RE = re.compile(r"Iteration\s+(-?\d+)\s+took\s+([0-9.eE+-]+)s")
LEN_RE = re.compile(r"Alignment Length:\s*(\d+)")
IDENT_RE = re.compile(r"Identity Score:\s*([0-9.]+)%")


def run_one(dataset, mode, version, timeout):
    cmd = ["./bin/main", dataset, str(mode), str(version)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s", file=sys.stderr)
        return None
    if p.returncode != 0:
        print(f"    exit {p.returncode}: {p.stderr.strip()[:200]}", file=sys.stderr)
        return None

    out = p.stdout
    align_len = LEN_RE.search(out)
    identity = IDENT_RE.search(out)
    iters = [(int(i), float(t)) for i, t in ITER_RE.findall(out)]
    if not iters:
        print("    no timing lines found", file=sys.stderr)
        return None

    return {
        "align_len": int(align_len.group(1)) if align_len else -1,
        "identity": float(identity.group(1)) if identity else -1.0,
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--datasets", nargs="+",
                    default=["150_10_small", "500_10", "150_10"])
    ap.add_argument("--cpu", nargs="*", type=int, default=[0, 1, 2, 3],
                    help="CPU versions to run (mode 0), empty to skip")
    ap.add_argument("-g", "--gpu", nargs="*", type=int, default=[0, 2, 3, 4, 5, 7],
                    help="GPU versions to run (mode 1), empty to skip")
    ap.add_argument("--hybrid", nargs="*", type=int, default=[0, 1, 2, 3],
                    help="hybrid versions to run (mode 2), empty to skip")
    ap.add_argument("--nocopy", nargs="*", type=int, default=[0, 2, 6],
                    help="no-copy GPU versions to run (mode 3), empty to skip")
    ap.add_argument("-o", "--out", default="outputs/timings.csv")
    ap.add_argument("-t", "--timeout", type=int, default=3600)
    args = ap.parse_args()

    if not os.path.isdir("datasets"):
        sys.exit("run me from the repository root")
    if not os.path.exists("./bin/main"):
        sys.exit("./bin/main not found - run make first")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    jobs = [(0, v, CPU_LABELS.get(v, str(v))) for v in args.cpu] + \
           [(1, v, GPU_LABELS.get(v, str(v))) for v in args.gpu] + \
           [(2, v, HYBRID_LABELS.get(v, str(v))) for v in args.hybrid] + \
           [(3, v, NOCOPY_LABELS.get(v, str(v))) for v in args.nocopy]

    for dataset in args.datasets:
        for mode, version, label in jobs:
            print(f"  {dataset:>16}  mode={mode} v{version} ({label})")
            res = run_one(dataset, mode, version, args.timeout)
            if res is None:
                continue
            for it, t in res["iters"]:
                rows.append({"dataset": dataset, "mode": mode, "version": version,
                             "label": label, "iter": it, "time_s": t,
                             "align_len": res["align_len"], "identity": res["identity"]})
            timed = [t for i, t in res["iters"] if i >= 0]
            if timed:
                print(f"      mean {sum(timed) / len(timed):.6f}s   "
                      f"len={res['align_len']} identity={res['identity']}%")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "mode", "version", "label", "iter",
                                          "time_s", "align_len", "identity"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
