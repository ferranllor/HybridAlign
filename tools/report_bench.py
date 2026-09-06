#!/usr/bin/env python3
"""
Every measurement the report needs, from one command, tagged with the machine it ran on.

    python3 tools/report_bench.py                          # full sweep, machine auto detected
    python3 tools/report_bench.py -m spark                 # force the machine label
    python3 tools/report_bench.py --quick                  # smoke test on the small datasets
    python3 tools/report_bench.py --skip batch             # drop one of the four parts

The point of the machine label is that the report compares a unified memory machine against a
discrete one, so every row carries where it came from and the figures can put the two side by side.
Run this once on the DGX Spark (-m spark) and once on the A100 box (-m a100), copy both output
directories together, and tools/report_figs.py draws them in one figure.

Four parts, each written to its own CSV under <out>/<machine>_*.csv:

  timings   wall time per (dataset, mode, version)      -> figure 2, table 1
  phases    hybrid GPU head / CPU tail split            -> figure 3a
  batch     multi read throughput vs batch size         -> figure 3b
  levels    per level kernel time per version, nsys      -> figures 1b and 3

plus <machine>_machine.json with what the machine is, so a figure can label itself.

Run from the repository root, with nothing else using the GPU.
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys

# The narrative order of the report: CPU first, then GPU, then the last column CPU baseline that
# makes the GPU comparison fair, then the hybrid. Kept as a list of (mode, version, label, group)
# so both the CSV and the figures inherit the same ordering and never re-sort by value.
PLAN = [
    (0, 0,  "cpu_sequential",     "cpu"),
    (0, 1,  "cpu_simd",           "cpu"),
    (0, 3,  "cpu_parallel_node",  "cpu"),
    (1, 0,  "gpu_naive",          "gpu"),
    (1, 5,  "gpu_async_batching", "gpu"),
    (1, 7,  "gpu_shared_mem",     "gpu"),
    (1, 8,  "gpu_last_col",       "gpu"),
    (1, 9,  "gpu_warps",          "gpu"),
    (1, 10, "gpu_registers",      "gpu"),
    (0, 4,  "cpu_last_col",       "fair"),
    (2, 4,  "hybrid_warps",       "hybrid"),
    (2, 5,  "hybrid_registers",   "hybrid"),
]

# Versions past the two that the report actually argues about. Off by default: they are a fair bit
# of extra runtime and the report only mentions them if there is room left for the paragraph.
EXTRA_PLAN = [
    (1, 12, "gpu_merged_req",     "gpu"),
    (1, 13, "gpu_short2",         "gpu"),
    (2, 6,  "hybrid_merged_req",  "hybrid"),
    (2, 7,  "hybrid_short2",      "hybrid"),
]

# Mode 3 is the same GPU kernels on a graph both processors address directly, which only means
# anything where the memory is physically shared. Worth having on the Spark, pointless on a
# discrete card, so it is opt in.
NOCOPY_PLAN = [
    (3, 7,  "nocopy_last_col",    "nocopy"),
    (3, 8,  "nocopy_warps",       "nocopy"),
    (3, 9,  "nocopy_registers",   "nocopy"),
]

DEFAULT_DATASETS = ["150_10", "brca2_150", "brca2_1500"]
QUICK_DATASETS = ["150_10_small", "brca2_150"]

# Batch sizes for mode 4. Powers of two up to where the per read time stops falling, which is the
# number the report compares against a real multi read aligner.
DEFAULT_BATCH = [1, 4, 16, 64, 256]

ITER_RE = re.compile(r"Iteration\s+(-?\d+)\s+took\s+([0-9.eE+-]+)s")
LEN_RE = re.compile(r"Alignment Length:\s*(\d+)")
IDENT_RE = re.compile(r"Identity Score:\s*([0-9.]+)%")
# [hybrid registers] total 134.71 ms | launch 0.03 ms (13 levels) | wait 40.99 ms (1 syncs) | ...
PHASE_RE = re.compile(
    r"\[hybrid (\S+)\] total\s+([0-9.]+) ms \| launch\s+([0-9.]+) ms \((\d+) levels\) \| "
    r"wait\s+([0-9.]+) ms \((\d+) syncs\) \| cpu\s+([0-9.]+) ms \((\d+) levels\) \| "
    r"traceback\s+([0-9.]+) ms")


# -------------------------------------------------------------------------------------------------
#                                          Machine
# -------------------------------------------------------------------------------------------------

def describe_machine():
    """What is running this, so a figure can title itself without the label being retyped."""
    info = {"host": platform.node(), "arch": platform.machine(), "gpu": "unknown", "sms": 0,
            "cpu": platform.processor() or "unknown", "cores": os.cpu_count()}

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"] = out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    return info


def auto_label(info):
    """A short slug for filenames, guessed from the GPU so -m can usually be left off."""
    gpu = info["gpu"].lower()
    for needle, slug in (("gb10", "spark"), ("a100", "a100"), ("h100", "h100"),
                         ("5060", "rtx5060ti"), ("4090", "rtx4090")):
        if needle in gpu:
            return slug
    return re.sub(r"[^a-z0-9]+", "", gpu.replace("nvidia", "")) or "unknown"


# -------------------------------------------------------------------------------------------------
#                                          One run
# -------------------------------------------------------------------------------------------------

def run_one(dataset, mode, version, timeout, env_extra=None):
    """One bin/main invocation -> its timed iterations, its alignment, its hybrid phase lines."""
    env = dict(os.environ)
    env["HYBRID_STATS"] = "1"       # harmless everywhere else, mode 2 prints its breakdown
    if env_extra:
        env.update(env_extra)

    cmd = ["./bin/main", dataset, str(mode), str(version)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"      TIMEOUT after {timeout}s", file=sys.stderr)
        return None
    if p.returncode != 0:
        print(f"      exit {p.returncode}: {p.stderr.strip()[:160]}", file=sys.stderr)
        return None

    iters = [(int(i), float(t)) for i, t in ITER_RE.findall(p.stdout)]
    if not iters:
        print("      no timing lines", file=sys.stderr)
        return None

    align_len = LEN_RE.search(p.stdout)
    identity = IDENT_RE.search(p.stdout)

    return {
        "iters": iters,
        "align_len": int(align_len.group(1)) if align_len else -1,
        "identity": float(identity.group(1)) if identity else -1.0,
        "phases": PHASE_RE.findall(p.stderr),
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# -------------------------------------------------------------------------------------------------
#                                          The parts
# -------------------------------------------------------------------------------------------------

def part_timings(machine, datasets, plan, timeout):
    """Wall time per implementation, and the hybrid phase breakdown that comes free with it."""
    timing_rows, phase_rows = [], []

    for dataset in datasets:
        print(f"\n  == {dataset} ==")
        for mode, version, label, group in plan:
            print(f"    {label:<20} (mode {mode} v{version}) ", end="", flush=True)
            res = run_one(dataset, mode, version, timeout)
            if res is None:
                continue

            for it, t in res["iters"]:
                timing_rows.append({"machine": machine, "dataset": dataset, "mode": mode,
                                    "version": version, "label": label, "group": group,
                                    "iter": it, "time_s": t,
                                    "align_len": res["align_len"], "identity": res["identity"]})

            timed = [t for i, t in res["iters"] if i >= 0]
            print(f"{mean(timed) * 1e3:8.2f} ms   identity {res['identity']}%")

            # The first phase line belongs to the verification run, before the caches are warm.
            for kernel, total, launch, glv, wait, syncs, cpu, clv, tb in res["phases"][1:]:
                phase_rows.append({"machine": machine, "dataset": dataset, "mode": mode,
                                   "version": version, "label": label, "kernel": kernel,
                                   "total_ms": float(total), "launch_ms": float(launch),
                                   "gpu_levels": int(glv), "wait_ms": float(wait),
                                   "syncs": int(syncs), "cpu_ms": float(cpu),
                                   "cpu_levels": int(clv), "traceback_ms": float(tb)})

    return timing_rows, phase_rows


def part_batch(machine, dataset, sizes, versions, timeout):
    """Mode 4: one graph, many reads. The per read time is what a real aligner is judged on."""
    rows = []

    print(f"\n  == batch sweep on {dataset} ==")
    for version in versions:
        for reads in sizes:
            print(f"    multi v{version} NUM_READS={reads:<5} ", end="", flush=True)
            res = run_one(dataset, 4, version, timeout, {"NUM_READS": str(reads)})
            if res is None:
                continue

            timed = [t for i, t in res["iters"] if i >= 0]
            per_read = mean(timed) / reads

            rows.append({"machine": machine, "dataset": dataset, "version": version,
                         "num_reads": reads, "time_s": mean(timed),
                         "per_read_ms": per_read * 1e3})
            print(f"{mean(timed) * 1e3:9.2f} ms total   {per_read * 1e3:7.3f} ms/read")

    return rows


def part_levels(machine, dataset, mode, version, label, outdir, timeout):
    """Per level kernel time from nsys, joined with the level widths by tools/report_figs.py.

    Run this over the whole GPU ladder, not one version. A GPU only version launches a kernel for
    *every* level, so one profile splits into the levels the hybrid would have kept - where a kernel
    has enough nodes to fill the machine, and where each optimisation has to prove itself - and the
    thin ones it would have handed to the CPU, where the same kernels collapse. Comparing those two
    halves across versions is what shows that the warp and register kernels are better at the job
    they were written for even where they lose end to end.

    bin/main aligns several times per process, so only the last run's launches are kept: those are
    the ones with warm caches, and they are the ones the timings above are the mean of."""
    if shutil.which("nsys") is None:
        print("    nsys not found, skipping the per level part", file=sys.stderr)
        return []

    stem = os.path.join(outdir, f"{machine}_{dataset}_{mode}.{version}")
    sqlite_path = stem + ".sqlite"
    env = dict(os.environ)
    env["HYBRID_STATS"] = "1"

    print(f"\n  == per level profile of {dataset}: {label} (mode {mode} v{version}) ==")
    try:
        subprocess.run(["nsys", "profile", "-t", "cuda", "-o", stem, "--force-overwrite", "true",
                        "./bin/main", dataset, str(mode), str(version)],
                       capture_output=True, text=True, timeout=timeout, env=env, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"    nsys failed: {exc}", file=sys.stderr)
        return []

    # nsys writes the sqlite next to the report the first time it is asked for a stats table -- and
    # on later runs it SILENTLY REUSES that file instead of re-exporting, so profiling into a
    # directory that already has one hands back the previous run's timings with no warning at all.
    # Delete it first and pass --force-export, or a re-measurement is not a measurement.
    try:
        os.remove(sqlite_path)
    except OSError:
        pass

    try:
        subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum",
                        "--format", "csv", stem + ".nsys-rep"],
                       capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        pass

    if not os.path.exists(sqlite_path):
        print("    no sqlite produced, skipping", file=sys.stderr)
        return []

    import sqlite3
    con = sqlite3.connect(sqlite_path)
    try:
        launches = con.execute(
            "SELECT start, end, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    except sqlite3.Error as exc:
        print(f"    could not read the kernel table: {exc}", file=sys.stderr)
        return []
    finally:
        con.close()

    if not launches:
        return []

    # Every alignment launches the same number of kernels, so the run length is the total divided
    # by how many alignments the process did: the verification run plus WARMUP plus NITER, which is
    # one more than the number of "Iteration" lines. A hybrid version says so directly.
    res = run_one(dataset, mode, version, timeout)
    per_run = len(launches)

    if res and res["phases"]:
        per_run = int(res["phases"][-1][3])                  # gpu_levels
    elif res and res["iters"]:
        per_run = max(1, len(launches) // (len(res["iters"]) + 1))

    last = launches[-per_run:] if per_run <= len(launches) else launches

    rows = []
    for i, (start, end, grid) in enumerate(last):
        rows.append({"machine": machine, "dataset": dataset, "mode": mode, "version": version,
                     "label": label, "level": i, "grid": grid, "ms": (end - start) / 1e6})

    print(f"    {len(rows)} levels, {sum(r['ms'] for r in rows):.1f} ms of kernel time")
    for r in rows[:6]:
        print(f"      level {r['level']:>4}  grid {r['grid']:>6}  {r['ms']:8.3f} ms")
    if len(rows) > 6:
        print(f"      ... and {len(rows) - 6} more")

    return rows


# -------------------------------------------------------------------------------------------------
#                                            Main
# -------------------------------------------------------------------------------------------------

def write_csv(path, rows, fields):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows):>5} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--machine", default=None,
                    help="label for this machine (default: guessed from the GPU name)")
    ap.add_argument("-d", "--datasets", nargs="+", default=None)
    ap.add_argument("-o", "--outdir", default="outputs/report")
    ap.add_argument("-t", "--timeout", type=int, default=3600)
    ap.add_argument("--quick", action="store_true",
                    help="small datasets only, for checking the script itself")
    ap.add_argument("--extra", action="store_true",
                    help="also run merged_req and short2")
    ap.add_argument("--nocopy", action="store_true",
                    help="also run mode 3, only meaningful where the memory is physically shared")
    ap.add_argument("--batch-dataset", default="150_10")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--batch-versions", nargs="+", type=int, default=[0])
    ap.add_argument("--levels-of", nargs="+",
                    default=["1:7", "1:8", "1:9", "1:10", "1:12", "1:13"],
                    help="mode:version list to profile per level with nsys. The default is the "
                         "whole GPU ladder: each is a GPU only version, so it launches a kernel "
                         "for every level, and the profile splits into the wide levels the hybrid "
                         "keeps and the thin ones it hands to the CPU")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["timings", "phases", "batch", "levels"],
                    help="parts to leave out")
    args = ap.parse_args()

    if not os.path.isdir("datasets"):
        sys.exit("run me from the repository root")
    if not os.path.exists("./bin/main"):
        sys.exit("./bin/main not found - run make first")

    info = describe_machine()
    machine = args.machine or auto_label(info)
    datasets = args.datasets or (QUICK_DATASETS if args.quick else DEFAULT_DATASETS)

    plan = list(PLAN)
    if args.extra:
        plan += EXTRA_PLAN
    if args.nocopy:
        plan += NOCOPY_PLAN

    os.makedirs(args.outdir, exist_ok=True)

    print(f"machine : {machine}")
    print(f"gpu     : {info['gpu']}")
    print(f"cpu     : {info['cpu']} ({info['cores']} procs, {info['arch']})")
    print(f"datasets: {' '.join(datasets)}")

    with open(os.path.join(args.outdir, f"{machine}_machine.json"), "w") as f:
        json.dump(dict(info, machine=machine), f, indent=2)

    if "timings" not in args.skip:
        timing_rows, phase_rows = part_timings(machine, datasets, plan, args.timeout)
        print()
        write_csv(os.path.join(args.outdir, f"{machine}_timings.csv"), timing_rows,
                  ["machine", "dataset", "mode", "version", "label", "group", "iter",
                   "time_s", "align_len", "identity"])
        if "phases" not in args.skip:
            write_csv(os.path.join(args.outdir, f"{machine}_phases.csv"), phase_rows,
                      ["machine", "dataset", "mode", "version", "label", "kernel", "total_ms",
                       "launch_ms", "gpu_levels", "wait_ms", "syncs", "cpu_ms", "cpu_levels",
                       "traceback_ms"])

    if "batch" not in args.skip:
        batch_rows = part_batch(machine, args.batch_dataset, args.batch_sizes,
                                args.batch_versions, args.timeout)
        print()
        write_csv(os.path.join(args.outdir, f"{machine}_batch.csv"), batch_rows,
                  ["machine", "dataset", "version", "num_reads", "time_s", "per_read_ms"])

    if "levels" not in args.skip:
        # Only the first dataset: the per level story is about one graph's shape, and profiling
        # every version on every dataset would multiply the runtime for nothing.
        ds = args.datasets[0] if args.datasets else "150_10"
        known = {(m, v): lab for m, v, lab, _ in PLAN + EXTRA_PLAN + NOCOPY_PLAN}

        level_rows = []
        for spec in args.levels_of:
            mode, version = (int(x) for x in spec.split(":"))
            label = known.get((mode, version), f"mode{mode}_v{version}")
            level_rows += part_levels(machine, ds, mode, version, label, args.outdir, args.timeout)

        print()
        write_csv(os.path.join(args.outdir, f"{machine}_levels.csv"), level_rows,
                  ["machine", "dataset", "mode", "version", "label", "level", "grid", "ms"])

    print(f"\ndone. Draw it with:  python3 tools/report_figs.py -i {args.outdir}")


if __name__ == "__main__":
    main()
