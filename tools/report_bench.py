#!/usr/bin/env python3
"""
Every measurement the report needs, from one command, tagged with the machine it ran on.

    python3 tools/report_bench.py                          # full sweep, machine auto detected
    python3 tools/report_bench.py -m spark                 # force the machine label
    python3 tools/report_bench.py --quick                  # smoke test on the small datasets
    python3 tools/report_bench.py --skip batch             # drop one of the four parts

The point of the machine label is that the report compares a unified memory machine against a
discrete one, so every row carries where it came from and the figures can put the two side by side.
Run this once on the DGX Spark (-m spark) and once on the A30 box (-m a30), copy both output
directories together, and tools/report_figs.py draws them in one figure.

Four parts, each written to its own CSV under <out>/<machine>_*.csv:

  timings   wall time per (dataset, mode, version)      -> figure 2, table 1
  phases    hybrid GPU head / CPU tail split            -> figure 3a
  batch     multi read throughput vs batch size         -> figure 3b
  levels    per level kernel time per version, nsys      -> figures 1b and 3
  hga       HGA vs our GPU batch vs our CPU batch, GCUPS -> state of the art comparison

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
    (2, 6,  "hybrid_merged_req",  "hybrid"),
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
# number the report compares against a real multi read aligner. 1024 reads of 150_10 is about
# 15 GB of last columns on each side, which a 24 GB A30 takes and a 128 GB GB10 barely notices;
# batch_fits() below trims the sweep to whatever the card in front of it can actually hold.
DEFAULT_BATCH = [1, 4, 16, 64, 256, 1024]

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
    for needle, slug in (("gb10", "spark"), ("a30", "a30"), ("a100", "a100"),
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


def batch_bytes_per_read(dataset):
    """Device bytes one extra read costs in mode 4: one last column per (node, read).

    init_gpu_graph_multi allocates num_nodes * num_reads * (M + 1) DTYPEMATRIX on the device, and
    init_cpu_graph_multi allocates the same again in pinned host memory. On a discrete card only
    the first competes for VRAM; on GB10 both come out of the one pool, which is why the caller
    doubles it there."""
    path = os.path.join("datasets", "graphs", "old", dataset + ".graph")
    seq_path = os.path.join("datasets", "sequences", "old", "S_" + dataset + ".seq")

    nodes = 0
    try:
        with open(path) as f:
            for line in f:
                if line[0] == "S":
                    nodes += 1
    except OSError:
        return 0

    m = 0
    try:
        with open(seq_path) as f:
            parts = f.readline().split()
            if len(parts) >= 3:
                m = len(parts[2])
    except OSError:
        pass

    return nodes * (m + 1) * 2          # sizeof(DTYPEMATRIX)


def batch_fits(sizes, dataset, unified):
    """Trim the sweep to the batches this machine can hold, rather than dying at the biggest one.

    A failed allocation in the middle of a sweep loses the whole run, and the largest batch is the
    one the report quotes, so it is worth spending a few lines to find out first."""
    per_read = batch_bytes_per_read(dataset)
    if per_read <= 0:
        return sizes

    free = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            free = int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    if free <= 0:
        return sizes

    # Both sides come out of one pool on a unified part; leave a quarter of the card spare either
    # way for the graph itself, the reads and the driver.
    budget = free * 0.75 / (2 if unified else 1)
    cap = int(budget // per_read)

    kept = [r for r in sizes if r <= cap]
    dropped = [r for r in sizes if r > cap]

    if dropped:
        print(f"    note: {per_read / 1e6:.1f} MB per read, {free / 1e9:.1f} GB free -> dropping "
              f"{', '.join(str(d) for d in dropped)} (cap {cap} reads)", file=sys.stderr)

    return kept or sizes[:1]


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


def part_hga(machine, datasets, sizes, solo_sizes, hga_bin, grid, block, outdir,
             timeout):
    """Three engines on the same graph and the same reads: HGA, our mode 4, our CPU mode 0 v5.

    Only the real graph. HGA stores each in-edge as an 8-bit distance from the vertex that reads
    it, so an in-neighbour more than 255 vertices back cannot be represented; cactus-BRCA2 peaks at
    92 and fits, 150_10 reaches 11.5 million and does not. That is a property of their data
    structure, not of their kernel, and pretending otherwise by feeding it a graph it cannot hold
    would not be a comparison.

    The common unit is GCUPS, which is what their paper reports and what
    "num_v * read_len * num_reads / seconds" comes to on both sides. It is the same quantity as our
    Gcell/s: HGA's num_v counts one vertex per base, and ours counts the same bases as node
    sequence, so a cell there is a cell here.

    solo_sizes are batches run for HGA alone. HGA gives one read to one thread and loops
    "read_id += grid * block", so it does not fill the GPU until the batch reaches grid * block
    reads -- 8704 at the suggested 68 x 128 -- and its DP buffers are sized by that thread count
    rather than by the batch, so a huge batch costs it no extra memory. Ours keeps a last column
    per (node, read), so the same batch would be tens of GB and simply will not fit. Running the
    big batches for HGA only is therefore what gives HGA its best case, and the honest way to
    quote it: their peak, against ours at a batch that fits."""
    rows = []

    if not os.path.exists(hga_bin):
        print(f"\n  == hga: {hga_bin} not built, running ours only (tools/hga_setup.sh) ==",
              file=sys.stderr)
        hga_bin = None

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import hga_convert
    except ImportError:
        print("  == hga: tools/hga_convert.py not importable, skipping ==", file=sys.stderr)
        return rows

    workdir = os.path.join(outdir, "hga_input")
    os.makedirs(workdir, exist_ok=True)

    for dataset in datasets:
        print(f"\n  == hga comparison on {dataset} ==")

        try:
            seqs, links, query = hga_convert.read_dataset(dataset)
        except (OSError, SystemExit):
            print(f"    cannot read {dataset}, skipped", file=sys.stderr)
            continue

        order = hga_convert.topological_order(seqs, links)
        bases, in_lists = hga_convert.build(seqs, links, order)
        num_v, m = len(bases), len(query)

        gpath = os.path.join(workdir, f"{dataset}.hga.graph")
        if not os.path.exists(gpath):
            hga_convert.write_graph(gpath, bases, in_lists)

        def gcups(reads, seconds):
            return num_v * m * reads / seconds / 1e9 if seconds > 0 else float("nan")

        print(f"    {num_v} vertices, {m} bp reads, HGA at {grid} x {block} "
              f"= {grid * block} threads")
        print(f"    {'reads':>7}{'HGA':>12}{'gpu_multi':>12}{'cpu_multi':>12}   (GCUPS)")

        def run_hga(reads):
            """-> (seconds, gcups) or None."""
            if not hga_bin:
                return None
            rpath = os.path.join(workdir, f"{dataset}.r{reads}.hga.reads")
            if not os.path.exists(rpath):
                hga_convert.write_reads(rpath, query, reads)
            # Match our scoring: +1 match, -1 mismatch, -1 gap. HGA negates mis and gap itself.
            cmd = [hga_bin, "-g", gpath, "-r", rpath, "-m", "1", "-n", "1", "-o", "1",
                   "-b", str(grid), "-t", str(block), "-d", "1"]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except (subprocess.SubprocessError, OSError):
                return None
            hit = re.search(r"Time:\s*([0-9.eE+-]+)s,\s*GCUPS:\s*([0-9.eE+-]+)", out.stdout)
            return (float(hit.group(1)), float(hit.group(2))) if hit else None

        for reads in sizes:
            line = {"machine": machine, "dataset": dataset, "read_len": m, "num_reads": reads}
            shown = {}

            got = run_hga(reads)
            if got:
                rows.append(dict(line, engine="hga", time_s=got[0], gcups=got[1],
                                 hga_threads=grid * block))
                shown["hga"] = got[1]

            # Ours: mode 4 is the GPU batch, mode 0 version 5 is the same batch on the CPU.
            for engine, mode, version in (("gpu_multi", 4, 0), ("cpu_multi", 0, 5)):
                res = run_one(dataset, mode, version, timeout, {"NUM_READS": str(reads)})
                if res is None:
                    continue
                t = mean([x for i, x in res["iters"] if i >= 0])
                rows.append(dict(line, engine=engine, time_s=t, gcups=gcups(reads, t)))
                shown[engine] = gcups(reads, t)

            def col(k):
                return f"{shown[k]:.2f}" if k in shown else "-"

            print(f"    {reads:>7}{col('hga'):>12}{col('gpu_multi'):>12}{col('cpu_multi'):>12}")

        # HGA alone, out where it finally has a read per thread and ours would not fit.
        solo = sorted(r for r in set(solo_sizes) | {grid * block} if r > max(sizes, default=0))
        for reads in solo:
            got = run_hga(reads)
            if not got:
                continue
            rows.append({"machine": machine, "dataset": dataset, "read_len": m,
                         "num_reads": reads, "engine": "hga", "time_s": got[0],
                         "gcups": got[1], "hga_threads": grid * block})
            note = "  <- one read per thread" if reads >= grid * block else ""
            print(f"    {reads:>7}{got[1]:>12.2f}{'-':>12}{'-':>12}{note}")

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
                    help="also run merged_req")
    ap.add_argument("--nocopy", action="store_true",
                    help="also run mode 3, only meaningful where the memory is physically shared")
    ap.add_argument("--batch-dataset", default="150_10")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--batch-versions", nargs="+", type=int, default=[0],
                    help="mode 4 versions: 0 = shared mem core, 1 = registers core")
    ap.add_argument("--no-batch-cap", action="store_true",
                    help="do not trim the batch sweep to what the card can hold")
    ap.add_argument("--hga-datasets", nargs="+", default=["brca2_150", "brca2_1500"],
                    help="graphs for the HGA comparison. Real graphs only: HGA's 8-bit in-edge "
                         "distance cannot represent 150_10 (see part_hga)")
    ap.add_argument("--hga-sizes", nargs="+", type=int, default=[1, 16, 64, 256, 1024],
                    help="reads per batch for the three-way comparison")
    ap.add_argument("--hga-solo-sizes", nargs="+", type=int, default=[4096, 8704, 17408],
                    help="batches run for HGA alone, past where ours fits in VRAM. HGA needs "
                         "grid*block reads before every thread has one, so this is where it "
                         "reaches its peak; grid*block is always added")
    ap.add_argument("--hga-grid", type=int, default=68,
                    help="HGA thread blocks (default 68, the value its README suggests and the "
                         "one the existing measurements used)")
    ap.add_argument("--hga-block", type=int, default=128, help="HGA threads per block")
    ap.add_argument("--hga-bin", default=os.path.join("outputs", "hga", "hga-src", "bin", "hga"),
                    help="HGA binary; build it with tools/hga_setup.sh")
    ap.add_argument("--levels-of", nargs="+",
                    default=["1:7", "1:8", "1:9", "1:10", "1:12"],
                    help="mode:version list to profile per level with nsys. The default is the "
                         "whole GPU ladder: each is a GPU only version, so it launches a kernel "
                         "for every level, and the profile splits into the wide levels the hybrid "
                         "keeps and the thin ones it hands to the CPU")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["timings", "phases", "batch", "levels", "hga"],
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
        # "unified" is the property that matters, not the vendor name: on GB10 the host and device
        # allocations come out of the same pool, so a batch costs twice what it costs on a card
        # with its own VRAM.
        unified = machine == "spark" or "gb10" in info["gpu"].lower()
        sizes = args.batch_sizes if args.no_batch_cap else batch_fits(
            args.batch_sizes, args.batch_dataset, unified)
        batch_rows = part_batch(machine, args.batch_dataset, sizes,
                                args.batch_versions, args.timeout)
        print()
        write_csv(os.path.join(args.outdir, f"{machine}_batch.csv"), batch_rows,
                  ["machine", "dataset", "version", "num_reads", "time_s", "per_read_ms"])

    if "hga" not in args.skip:
        hga_rows = part_hga(machine, args.hga_datasets, args.hga_sizes, args.hga_solo_sizes,
                            args.hga_bin, args.hga_grid, args.hga_block, args.outdir,
                            args.timeout)
        print()
        write_csv(os.path.join(args.outdir, f"{machine}_hga.csv"), hga_rows,
                  ["machine", "dataset", "engine", "read_len", "num_reads", "time_s", "gcups",
                   "hga_threads"])

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
