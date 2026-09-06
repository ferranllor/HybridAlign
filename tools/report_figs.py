#!/usr/bin/env python3
r"""
The three figures of the report, plus the numbers its prose quotes, from whatever
tools/report_bench.py left in the input directory.

    python3 tools/report_graph_stats.py                # once, machine independent
    python3 tools/report_bench.py -m spark             # on the DGX Spark
    python3 tools/report_bench.py -m a30               # on the discrete box
    python3 tools/report_figs.py                       # -> report/fig_*.pdf, report/numbers.tex

Whatever machines it finds, it draws. One machine gives one panel per figure, two give two, so the
same command works while only one of the two boxes has been run.

  fig_structure  how many nodes sit at each level of the two graphs, on its own
                 -> the shape of the input, which the whole design follows from

  fig_throughput what the GPU achieves as a level narrows, and where the CPU overtakes it
                 -> why the split is where it is

  fig_kernels    (a) throughput of each GPU rung on the levels the hybrid keeps
                 (b) what the same rung costs on the levels it gives away
                 -> every optimisation earned its keep, on the levels it was written for

  fig_runtime    wall time per implementation, CPU then GPU then the fair CPU baseline then hybrid
                 -> the result

  table_hga      HGA vs our GPU batch vs our CPU batch, GCUPS, as a LaTeX table
                 -> the state of the art comparison, best batch per engine in bold

numbers.tex is a set of \\newcommand definitions so the prose never quotes a number by hand;
report.tex reads it with \\input{numbers}.

Figures are sized from the report's own geometry (see COL_W / TEXT_W), so
\includegraphics[width=\columnwidth] does not rescale them.
Needs matplotlib:  pip install matplotlib
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    sys.exit("matplotlib is not installed: pip install matplotlib")

# Print palette: dark enough to survive a greyscale printout, and the four groups are also
# separated by their position in the fixed ordering, never by colour alone.
INK = "#111111"
INK2 = "#5a5a5a"
GRID = "#d8d8d4"
SURFACE = "#ffffff"

GROUP_COLOR = {
    "cpu":    "#e06c2a",   # orange
    "gpu":    "#2a6fd6",   # blue
    "fair":   "#8a5cd0",   # violet, the CPU baseline that makes the GPU comparison fair
    "hybrid": "#14926a",   # green
    "nocopy": "#c08a00",   # amber
}
GROUP_NAME = {"cpu": "CPU", "gpu": "GPU only", "fair": "CPU (last col.)",
              "hybrid": "hybrid", "nocopy": "no copy"}

DATASET_NAME = {"150_10": "synthetic 150_10", "brca2_150": "cactus BRCA2",
                "brca2_1500": "cactus BRCA2, 1.5 kb read"}

QUERY_LEN = {"150_10": 149, "150_10_small": 149, "brca2_150": 150,
             "brca2_400": 400, "brca2_1500": 1500, "brca2_4500": 4500, "500_10": 499}

# Figure widths in inches, taken from the report's own geometry rather than guessed: A4 with
# left=right=2cm gives a 17cm text block, and two columns with the default 10pt gutter give
# 8.32cm each. Matching these exactly means \includegraphics never rescales the figure, so the
# tick labels come out at the size matplotlib drew them.
COL_W = 3.290     # \columnwidth  = 236.85pt
TEXT_W = 6.718    # \textwidth    = 483.70pt

plt.rcParams.update({
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# -------------------------------------------------------------------------------------------------
#                                          Loading
# -------------------------------------------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_all(indir):
    """-> {machine: {"timings": [...], "phases": [...], "batch": [...], "levels": [...], "info": {}}}"""
    data = defaultdict(lambda: defaultdict(list))

    for path in sorted(glob.glob(os.path.join(indir, "*_timings.csv"))):
        machine = os.path.basename(path)[: -len("_timings.csv")]
        data[machine]["timings"] = read_csv(path)
        for part in ("phases", "batch", "levels", "hga"):
            data[machine][part] = read_csv(os.path.join(indir, f"{machine}_{part}.csv"))

        info_path = os.path.join(indir, f"{machine}_machine.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                data[machine]["info"] = json.load(f)

    return data


def machine_title(machine, info):
    """'spark' plus what it actually is, so a panel needs no caption of its own."""
    gpu = (info or {}).get("gpu", "").replace("NVIDIA ", "").replace("GeForce ", "")
    return f"{machine}" + (f"  ({gpu})" if gpu else "")


def stats(rows, key=lambda r: True, value=lambda r: float(r["time_s"])):
    """mean/min/max over the timed iterations of the rows that pass the filter."""
    vals = [value(r) for r in rows if int(r["iter"]) >= 0 and key(r)]
    if not vals:
        return None
    return sum(vals) / len(vals), min(vals), max(vals)


def fmt_ms(v):
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def clean_ax(ax, xgrid=True, ygrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=2.5)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# -------------------------------------------------------------------------------------------------
#                        Figure 1: the shape of the problem
# -------------------------------------------------------------------------------------------------

def fig_structure(levels_csv, outdir, show, width=COL_W):
    """Nodes per topological level: the shape of the input, and nothing else.

    This is the one plot the whole design follows from, so it stands alone. Levels are in
    topological order, which is half the message: the wide levels are not merely few, they are all
    at the front, so the GPU part and the CPU part of one alignment are strictly ordered and cannot
    overlap. The x axis is log(level + 1) so the dozen levels holding the parallelism are not
    squashed into the first pixel."""
    rows = read_csv(levels_csv)
    if not rows:
        print("  no graph_levels.csv, run tools/report_graph_stats.py first", file=sys.stderr)
        return

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append((int(r["level"]), int(r["width"]), int(r["bases"])))
    for ds in by_ds:
        by_ds[ds].sort()

    drawn = [ds for ds in show if ds in by_ds] or sorted(by_ds)

    fig, ax = plt.subplots(figsize=(width, width * 0.62))

    styles = {"150_10": (GROUP_COLOR["gpu"], 1.3, 0.20),
              "brca2_150": (GROUP_COLOR["cpu"], 1.1, 0.35)}
    for ds in drawn:
        pts = by_ds[ds]
        color, lw, alpha = styles.get(ds, (INK2, 1.0, 0.2))
        xs = [p[0] + 1 for p in pts]
        ys = [max(p[1], 1) for p in pts]
        ax.fill_between(xs, 1, ys, step="mid", color=color, alpha=alpha, lw=0, zorder=2)
        ax.step(xs, ys, where="mid", color=color, lw=lw,
                label=DATASET_NAME.get(ds, ds), zorder=3)

    ax.axhline(16, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, max(p[0] for ds in drawn for p in by_ds[ds]) + 1)
    ax.set_ylim(1, None)
    ax.set_xlabel("topological level (log, 1 = first)")
    ax.set_ylabel("nodes in the level")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, handlelength=1.2,
              borderpad=0.1, labelspacing=0.25)
    clean_ax(ax, xgrid=False, ygrid=True)

    fig.tight_layout(pad=0.3)
    save(fig, outdir, "fig_structure")


def fig_throughput(levels_csv, data, outdir, dataset, width=COL_W, cut=16):
    """Achieved throughput against level width: what it costs to ignore the shape above.

    Kept as its own figure now that fig_structure is density only. One point per width bin, from a
    profiled GPU-only run, so it covers the thin levels the hybrid never sends to the GPU. The
    horizontal line is what the CPU delivers on those same thin levels, and where the curve crosses
    it is where handing the level over starts to pay."""
    widths = {}
    for r in read_csv(levels_csv):
        if r["dataset"] == dataset:
            widths[int(r["level"])] = (int(r["width"]), int(r["bases"]))
    if not widths:
        print(f"  no level widths for {dataset}, skipping fig_throughput", file=sys.stderr)
        return

    m = QUERY_LEN.get(dataset, 149)

    def bins_of(w):
        edges = [1, 2, 3, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]
        for lo, hi in zip(edges, edges[1:]):
            if lo <= w < hi:
                return lo, hi
        return edges[-1], edges[-1] * 2

    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    drew = False

    for machine, d in sorted(data.items()):
        lv = [r for r in d.get("levels", []) if r["dataset"] == dataset and r.get("mode") == "1"]
        if not lv:
            continue
        # One representative kernel, so the figure is about level width and not about the ladder.
        pref = [l for l in ("gpu_warps", "gpu_last_col", "gpu_shared_mem")
                if any(r.get("label") == l for r in lv)]
        if pref:
            lv = [r for r in lv if r.get("label") == pref[0]]

        grouped = defaultdict(lambda: [0.0, 0.0])
        for r in lv:
            lvl, ms = int(r["level"]), float(r["ms"])
            if lvl not in widths or ms <= 0:
                continue
            w, b = widths[lvl]
            slot = grouped[bins_of(w)]
            slot[0] += b * m
            slot[1] += ms * 1e-3

        pts = sorted((lo * (hi / lo) ** 0.5, c / sec / 1e9)
                     for (lo, hi), (c, sec) in grouped.items() if sec > 0)
        if not pts:
            continue

        color = GROUP_COLOR["hybrid"] if machine == "spark" else GROUP_COLOR["gpu"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3.0, lw=1.2,
                color=color, label=f"{machine}, GPU", zorder=3)
        drew = True

        ph = [r for r in d.get("phases", []) if r["dataset"] == dataset]
        thin_cells = sum(b for _, (w, b) in widths.items() if w < cut) * m
        if ph and thin_cells:
            cpu_ms = sum(float(r["cpu_ms"]) for r in ph) / len(ph)
            rate = thin_cells / (cpu_ms * 1e-3) / 1e9
            ax.axhline(rate, color=GROUP_COLOR["cpu"], lw=1.1, ls=(0, (3, 2)), zorder=4)

    ax.axvline(cut, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes in the level (one warp each)")
    ax.set_ylabel("achieved  Gcell/s")
    if drew:
        ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    else:
        ax.text(0.5, 0.5, "run tools/report_bench.py\n(levels part needs nsys)",
                transform=ax.transAxes, ha="center", va="center", color=INK2, fontsize=7)
    clean_ax(ax, xgrid=False, ygrid=True)

    fig.tight_layout(pad=0.3)
    save(fig, outdir, "fig_throughput")


# -------------------------------------------------------------------------------------------------
#                 Figure: what each kernel bought, and where it bought it
# -------------------------------------------------------------------------------------------------

# The rungs of the GPU ladder, in the order they were written. Anything profiled that is not in
# here is still drawn, at the end, so adding a kernel needs no edit.
# Which half of the report a version belongs to. report_bench writes a "part" column, but CSVs
# collected before the split do not have one, so the label is used as a fallback and both vintages
# of data draw correctly.
LASTCOL_LABELS = {"cpu_last_col", "gpu_last_col", "gpu_warps", "gpu_registers", "gpu_merged_req",
                  "gpu_short2", "gpu_persistent_kernels", "hybrid_warps", "hybrid_registers",
                  "hybrid_merged_req", "hybrid_short2", "nocopy_last_col", "nocopy_warps",
                  "nocopy_registers", "nocopy_persistent_kernels"}


def part_of(row):
    if row.get("part"):
        return row["part"]
    return "lastcol" if row.get("label") in LASTCOL_LABELS else "matrix"


KERNEL_ORDER = ["gpu_shared_mem", "gpu_last_col", "gpu_warps", "gpu_registers",
                "gpu_merged_req"]

KERNEL_SHORT = {"gpu_shared_mem": "shared_mem", "gpu_last_col": "last_col",
                "gpu_warps": "warps", "gpu_registers": "registers",
                "gpu_merged_req": "merged_req"}


def kernel_split(levels_csv, data, dataset, cut=16, top=3):
    """-> ({machine: {label: (top-N Gcell/s, dense Gcell/s, thin Gcell/s)}}, cells per band).

    Three bands, all in the same unit so one axis carries them:

      top    the `top` widest levels. cut=16 is the scheduler's threshold, but it is a low bar for
             a kernel that wants thousands of nodes in flight, so the widest few levels are where
             a warp-per-node design is actually seen at its best.
      dense  every level of at least `cut` nodes, which is exactly what the hybrid sends to the GPU.
      thin   the rest, which the hybrid sends to the CPU.
    """
    widths = {}
    for r in read_csv(levels_csv):
        if r["dataset"] == dataset:
            widths[int(r["level"])] = (int(r["width"]), int(r["bases"]))
    if not widths:
        return {}, (0.0, 0.0, 0.0)

    m = QUERY_LEN.get(dataset, 149)
    top_levels = {lvl for lvl, _ in sorted(widths.items(), key=lambda kv: -kv[1][0])[:top]}

    cells = (sum(b for lvl, (w, b) in widths.items() if lvl in top_levels) * m,
             sum(b for w, b in widths.values() if w >= cut) * m,
             sum(b for w, b in widths.values() if w < cut) * m)

    out = {}
    for machine, d in data.items():
        acc = defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in d.get("levels", []):
            if r["dataset"] != dataset or r.get("mode") != "1":
                continue
            lvl, ms = int(r["level"]), float(r["ms"])
            if lvl not in widths or ms <= 0:
                continue
            a = acc[r.get("label", "")]
            if lvl in top_levels:
                a[0] += ms * 1e-3
            if widths[lvl][0] >= cut:
                a[1] += ms * 1e-3
            else:
                a[2] += ms * 1e-3

        per = {}
        for label, secs in acc.items():
            if all(x > 0 for x in secs):
                per[label] = tuple(c / t / 1e9 for c, t in zip(cells, secs))
        if per:
            out[machine] = per

    return out, cells


def cpu_baseline_gcups(data, machine, levels_csv, dataset):
    """The CPU last-column version's throughput over the whole graph, as the reference line.

    Whole graph rather than dense levels only: the CPU run is not instrumented per level, and
    quoting a dense-only CPU figure would mean inventing a split it never measured. It is the
    conservative direction anyway -- the thin levels drag this number down, so the GPU's advantage
    over it is if anything understated."""
    total = sum(int(r["bases"]) for r in read_csv(levels_csv) if r["dataset"] == dataset)
    total *= QUERY_LEN.get(dataset, 149)

    rows = [r for r in data.get(machine, {}).get("timings", []) if r["dataset"] == dataset]
    st = stats(rows, key=lambda r: r["label"] == "cpu_last_col")
    return total / st[0] / 1e9 if (st and total) else None


def fig_kernels(levels_csv, data, outdir, dataset, cut=16, top=3, width=TEXT_W):
    """Each GPU version's throughput on the widest few levels, on all dense levels, and on the
    thin ones, with the CPU last-column version as a reference line.

    Logarithmic, because the bands are two orders of magnitude apart on the same kernel; on a
    linear axis the thin bars would be invisible, which is the one thing the figure exists to
    show."""
    per_machine, _ = kernel_split(levels_csv, data, dataset, cut, top)
    if not per_machine:
        print(f"  no per level data for {dataset}, skipping fig_kernels", file=sys.stderr)
        return

    machines = sorted(per_machine)
    labels = [l for l in KERNEL_ORDER if any(l in per_machine[m] for m in machines)]
    labels += sorted({l for m in machines for l in per_machine[m]} - set(labels))

    bands = [(0, f"{top} widest levels", GROUP_COLOR["hybrid"]),
             (1, f"levels of $\\geq${cut} nodes", GROUP_COLOR["gpu"]),
             (2, f"levels of $<${cut} nodes", GROUP_COLOR["cpu"])]

    n = len(machines)
    fig, axes = plt.subplots(1, n, figsize=(width if n > 1 else COL_W, 0.30 * len(labels) + 1.10),
                             squeeze=False, sharex=True)

    vals = [v for m in machines for t in per_machine[m].values() for v in t]
    lo, hi = min(vals), max(vals)

    for ax, machine in zip(axes[0], machines):
        y = list(range(len(labels)))
        for idx, name, colour in bands:
            off = (idx - 1) * 0.26
            ax.barh([v + off for v in y],
                    [per_machine[machine].get(l, (0, 0, 0))[idx] for l in labels],
                    height=0.24, color=colour, zorder=3, label=name)

        base = cpu_baseline_gcups(data, machine, levels_csv, dataset)
        if base:
            ax.axvline(base, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4,
                       label="CPU last_col, whole graph")

        ax.set_yticks(y)
        ax.set_yticklabels([KERNEL_SHORT.get(l, l) for l in labels], color=INK2)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlim(lo / 3.0, hi * 3.0)
        ax.set_xlabel("Gcell/s (log)")
        clean_ax(ax)

    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), labelcolor=INK2, handlelength=1.1,
               columnspacing=1.2)

    fig.tight_layout(pad=0.35, rect=(0, 0.10, 1, 1))
    save(fig, outdir, "fig_kernels")


# -------------------------------------------------------------------------------------------------
#                        Figure 2: the result
# -------------------------------------------------------------------------------------------------

def fig_runtime(data, outdir, dataset, part="matrix", name="fig_runtime"):
    """Wall time per implementation, one panel per machine, in the report's narrative order."""
    machines = [m for m in sorted(data) if data[m].get("timings")]
    if not machines:
        print("  no timings to draw", file=sys.stderr)
        return

    def keep(r):
        return part is None or part_of(r) == part

    # A single fixed ordering across panels. Grouped by mode first, so every CPU row is adjacent
    # to every other CPU row and the colour never jumps around the axis: report_bench appends the
    # opt-in versions after the main plan, which would otherwise scatter them.
    order, seen = [], set()
    for m in machines:
        for r in data[m]["timings"]:
            if not keep(r):
                continue
            key = (r["label"], r["group"])
            if key not in seen:
                seen.add(key)
                order.append(key)

    if not order:
        print(f"  no '{part}' rows for {dataset}, skipping {name}", file=sys.stderr)
        return

    rank = {g: i for i, g in enumerate(("cpu", "gpu", "fair", "hybrid", "nocopy"))}
    seen_at = {kv: i for i, kv in enumerate(order)}          # snapshot: the sort mutates order
    order.sort(key=lambda kv: (rank.get(kv[1], len(rank)), seen_at[kv]))

    n = len(machines)
    fig, axes = plt.subplots(1, n, figsize=(TEXT_W if n > 1 else COL_W, 0.142 * len(order) + 0.88),
                             squeeze=False, sharey=True)

    for ax, machine in zip(axes[0], machines):
        rows = [r for r in data[machine]["timings"] if r["dataset"] == dataset and keep(r)]
        labels, values, los, his, colors = [], [], [], [], []

        for label, group in order:
            s = stats(rows, key=lambda r, l=label: r["label"] == l)
            if s is None:
                continue
            mean_s, lo, hi = s
            labels.append(label.split("_", 1)[1] if "_" in label else label)
            values.append(mean_s * 1e3)
            los.append(lo * 1e3)
            his.append(hi * 1e3)
            colors.append(GROUP_COLOR.get(group, INK2))

        if not values:
            ax.set_visible(False)
            continue

        y = list(range(len(values)))
        ax.barh(y, values, height=0.66, color=colors, zorder=3)
        for i, (lo, hi) in enumerate(zip(los, his)):
            if hi > lo:
                ax.plot([lo, hi], [i, i], color=INK, lw=0.8, zorder=4)

        best = min(values)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=INK2)
        ax.set_ylim(len(values) - 0.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlim(best * 0.55, max(values) * 1.35)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("wall time per alignment (ms, log)")
        clean_ax(ax)

    present = [g for g in ("cpu", "gpu", "fair", "hybrid", "nocopy")
               if any(g == gr for _, gr in order)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in present]
    names = [GROUP_NAME[g] for g in present]
    fig.legend(handles, names, frameon=False, ncol=len(names), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)

    fig.tight_layout(pad=0.4, rect=(0, 0.055, 1, 1))
    save(fig, outdir, name)


ENGINE_NAME = {"hga": "HGA", "gpu_multi": "ours, GPU batch", "cpu_multi": "ours, CPU batch"}
ENGINE_ORDER = ["hga", "gpu_multi", "cpu_multi"]


def table_hga(data, outdir, dataset):
    """The state-of-the-art comparison as a table rather than a plot.

    Three engines and a handful of batch sizes is a small enough grid that a table carries it
    without spending a figure on it, and a table can show the exact numbers a reader wants to
    quote. The best batch for each engine is bold, which is the whole point of the comparison:
    HGA needs thousands of reads to reach its peak because it gives one read to one thread."""
    machines = [m for m in sorted(data)
                if any(r["dataset"] == dataset for r in data[m].get("hga", []))]
    if not machines:
        print(f"  no hga rows for {dataset}, skipping table_hga", file=sys.stderr)
        return

    sizes, cells = [], {}
    for machine in machines:
        for r in data[machine]["hga"]:
            if r["dataset"] != dataset:
                continue
            reads = int(r["num_reads"])
            if reads not in sizes:
                sizes.append(reads)
            cells[(machine, r["engine"], reads)] = float(r["gcups"])
    sizes.sort()

    lines = ["% Generated by tools/report_figs.py -- do not edit, re-run the script.",
             "\\begin{tabular}{ll" + "r" * len(sizes) + "}",
             "\\hline",
             "System & Engine & " + " & ".join(f"{r:,}".replace(",", "\\,") for r in sizes)
             + " \\\\",
             "\\hline"]

    for machine in machines:
        gpu = (data[machine].get("info") or {}).get("gpu", machine)
        first = True
        for engine in ENGINE_ORDER:
            row = {r: cells[(machine, engine, r)] for r in sizes
                   if (machine, engine, r) in cells}
            if not row:
                continue
            # The smallest batch that reaches the plateau, not the largest measured: past
            # saturation the remaining points differ by noise, and bolding the last one would
            # claim an engine needs more reads than it does.
            top = max(row.values())
            best = min(r for r, g in row.items() if g >= 0.99 * top)
            out = []
            for r in sizes:
                if r not in row:
                    out.append("--")
                else:
                    txt = f"{row[r]:.2f}" if row[r] < 1 else f"{row[r]:.1f}"
                    out.append(f"\\textbf{{{txt}}}" if r == best else txt)
            lines.append((gpu if first else "") + " & " + ENGINE_NAME.get(engine, engine)
                         + " & " + " & ".join(out) + " \\\\")
            first = False
        lines.append("\\hline")

    lines.append("\\end{tabular}")

    path = os.path.join(outdir, "table_hga.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# -------------------------------------------------------------------------------------------------
#                        numbers.tex
# -------------------------------------------------------------------------------------------------

# A LaTeX command name can only hold letters, and the names here are built out of dataset and
# machine labels that are mostly digits, so stripping them collides: brca2_150 and brca2_1500 both
# become "Brca", and two \newcommand of one name is a build error rather than a wrong number.
# Curated tags keep the common cases readable; anything else spells its digits out, which is ugly
# but unique.
DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}

DATASET_TAG = {"150_10": "Syn", "150_10_small": "SynSmall", "500_10": "SynLong",
               "brca2_150": "Brca", "brca2_400": "BrcaFour", "brca2_1500": "BrcaLong",
               "brca2_4500": "BrcaXL"}

MACHINE_TAG = {"spark": "Spark", "a30": "Ampere", "a100": "AHundred", "h100": "Hopper",
               "rtx5060ti": "Blackwell", "rtx4090": "Ada"}


def tex_name(*parts):
    """A LaTeX command name: letters only, digits spelled so nothing collides.

    Each underscore separated word is capitalised, so cpu_last_col reads back as CpuLastCol."""
    out = []
    for p in parts:
        for word in re.split(r"[^0-9A-Za-z]+", p):
            letters = "".join(DIGIT_WORD.get(c, c) for c in word)
            if letters:
                out.append(letters[:1].upper() + letters[1:])
    return "".join(out)


def dataset_tag(ds):
    return DATASET_TAG.get(ds) or tex_name(ds)


def machine_tag(machine):
    return MACHINE_TAG.get(machine) or tex_name(machine)


def write_numbers(data, levels_csv, outdir, dataset):
    """Every number the prose quotes, as \\newcommand, so the text cannot drift from the data."""
    # Two things make this file safe to paste into a document that was not built around it.
    #
    # The unit macros: \SI is used throughout the prose, but siunitx is not in every preamble (and
    # \bp is not one of its units anyway), so a lightweight stand-in is defined here, guarded, and
    # skipped whenever the real package is already loaded.
    #
    # \providecommand + \renewcommand instead of a bare \newcommand: pasting this file into a
    # document that already has an older copy of it -- or pasting it twice -- would otherwise fail
    # with "command already defined", which is exactly the kind of error that is annoying to chase
    # in a web editor. This form always ends up with the current value and never errors.
    lines = ["% Generated by tools/report_figs.py - do not edit, re-run the script.",
             "% Paste or \\input this anywhere before the text that uses it. Safe to re-paste:",
             "% every macro is (re)defined rather than newly declared.",
             "",
             "% Minimal stand-in for siunitx, only if the real thing is not loaded.",
             "\\makeatletter",
             "\\@ifundefined{SI}{%",
             "  \\newcommand{\\SI}[2]{#1\\,#2}%",
             "  \\newcommand{\\giga}{G}\\newcommand{\\mega}{M}\\newcommand{\\kilo}{k}%",
             "  \\newcommand{\\milli}{m}\\newcommand{\\micro}{\\ensuremath{\\mu}}%",
             "  \\newcommand{\\byte}{B}\\newcommand{\\second}{s}%",
             "  \\newcommand{\\bp}{bp}\\newcommand{\\per}{/}%",
             "}{}",
             "\\makeatother",
             ""]
    defined = set()

    def define(name, value):
        if name in defined:
            print(f"  name clash on \\{name}, skipped (fix the tag map)", file=sys.stderr)
            return
        defined.add(name)
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}")

    # ---- graph structure, machine independent ---------------------------------------------------
    rows = read_csv(levels_csv)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append((int(r["width"]), int(r["bases"])))

    level_width = {int(r["level"]): (int(r["width"]), int(r["bases"]))
                   for r in rows if r["dataset"] == dataset}
    tot_cells = sum(b for _, b in level_width.values()) * QUERY_LEN.get(dataset, 149)

    for ds, pts in by_ds.items():
        tag = dataset_tag(ds)
        wide = [(w, b) for w, b in pts if w >= 16]
        total_bases = sum(b for _, b in pts)
        define(f"G{tag}Nodes", f"{sum(w for w, _ in pts):,}".replace(",", "\\,"))
        define(f"G{tag}Levels", f"{len(pts):,}".replace(",", "\\,"))
        define(f"G{tag}MaxWidth", f"{max(w for w, _ in pts):,}".replace(",", "\\,"))
        define(f"G{tag}WideLevels", str(len(wide)))
        define(f"G{tag}ThinLevels", f"{len(pts) - len(wide):,}".replace(",", "\\,"))
        define(f"G{tag}WidePct", f"{100.0 * sum(b for _, b in wide) / max(total_bases, 1):.0f}")
        define(f"G{tag}ThinPct",
               f"{100.0 * (total_bases - sum(b for _, b in wide)) / max(total_bases, 1):.0f}")
        define(f"G{tag}Gcells", f"{total_bases * QUERY_LEN.get(ds, 149) / 1e9:.2f}")

        # A level still costs its longest node even with a thread per node, so this is the ceiling
        # on parallelising the tail the CPU is given.
        thin_rows = [r for r in rows
                     if r["dataset"] == ds and int(r["width"]) < 16 and r.get("maxlen")]
        thin_work = sum(int(r["bases"]) for r in thin_rows)
        thin_path = sum(int(r["maxlen"]) for r in thin_rows)
        if thin_path and thin_work:
            define(f"G{tag}TailPathPct", f"{100.0 * thin_path / thin_work:.0f}")
            define(f"G{tag}TailMaxGain", f"{thin_work / thin_path:.2f}")

    lines.append("")

    # ---- per machine ----------------------------------------------------------------------------
    for machine, d in sorted(data.items()):
        mt = machine_tag(machine)
        info = d.get("info", {})
        define(f"{mt}Gpu", info.get("gpu", "?").replace("NVIDIA ", ""))
        define(f"{mt}Cpu", info.get("cpu", "?").split("@")[0].strip())
        define(f"{mt}Cores", str(info.get("cores", "?")))

        rows = [r for r in d.get("timings", []) if r["dataset"] == dataset]
        order_groups = sorted({(r["label"], r["group"]) for r in rows})
        parts = {r["label"]: part_of(r) for r in rows}
        best_label, best_ms = None, None

        for label in sorted({r["label"] for r in rows}):
            s = stats(rows, key=lambda r, l=label: r["label"] == l)
            if s is None:
                continue
            ms = s[0] * 1e3
            define(f"{mt}{tex_name(label)}", f"{ms:.0f}" if ms >= 10 else f"{ms:.1f}")
            if best_ms is None or ms < best_ms:
                best_label, best_ms = label, ms

        if best_ms is not None:
            define(f"{mt}Best", f"{best_ms:.0f}")
            define(f"{mt}BestName", best_label.replace("_", " "))

        base = stats(rows, key=lambda r: r["label"] == "cpu_sequential")
        if base and best_ms:
            define(f"{mt}Speedup", f"{base[0] * 1e3 / best_ms:.0f}")

        # Every ratio the prose quotes, derived here so none of them is typed by hand and none can
        # go stale when a machine is re-measured.
        def ms_of(label):
            st = stats(rows, key=lambda r, l=label: r["label"] == l)
            return st[0] * 1e3 if st else None

        for part_name, tag in (("matrix", "Matrix"), ("lastcol", "LastCol")):
            got = [(l, ms_of(l)) for l in parts if parts[l] == part_name]
            got = [(l, v) for l, v in got if v]
            if got:
                lab, val = min(got, key=lambda kv: kv[1])
                define(f"{mt}Best{tag}", f"{val:.0f}")
                define(f"{mt}Best{tag}Name", lab.replace("_", " "))

        # Each rung against the one before it, in the order the versions were written, so the
        # prose can quote "this bought Nx" without anybody typing a ratio.
        MATRIX_LADDER = ["gpu_naive", "gpu_parallel_node", "gpu_parallel_async",
                         "gpu_async_monolithic", "gpu_async_batching", "gpu_shared_mem"]
        prev = None
        for lab in MATRIX_LADDER:
            cur = ms_of(lab)
            if cur and prev:
                define(f"{mt}GainOf{tex_name(lab.split('_', 1)[1])}", f"{prev / cur:.1f}")
            if cur:
                prev = cur

        seq, simd = ms_of("cpu_sequential"), ms_of("cpu_simd")
        if seq and simd:
            define(f"{mt}SimdGain", f"{seq / simd:.1f}")

        pnode = ms_of("cpu_parallel_node")
        if pnode and best_ms:
            define(f"{mt}HybridOverParallelNode", f"{pnode / best_ms:.2f}")

        gpu_only = [v for v in (ms_of(l) for l, g in order_groups if g == "gpu") if v]
        if gpu_only and best_ms:
            define(f"{mt}BestGpuOnly", f"{min(gpu_only):.0f}")
            define(f"{mt}HybridOverBestGpu", f"{min(gpu_only) / best_ms:.1f}")

        # No-copy against the same kernel with explicit copies: what the shared graph is worth
        # to a GPU-only version, as opposed to what it is worth to the hybrid.
        nc = [(ms_of(f"nocopy_{k}"), ms_of(f"gpu_{k}")) for k in ("last_col", "warps", "registers")]
        nc = [(a, b) for a, b in nc if a and b]
        if nc:
            gains = [100.0 * (b - a) / b for a, b in nc]
            define(f"{mt}NocopyGainPct", f"{sum(gains) / len(gains):.0f}")

        # What is left for further kernel work once the CPU tail is fixed: Amdahl on the split.
        ph_all = [r for r in d.get("phases", []) if r["dataset"] == dataset]
        if ph_all:
            n = len(ph_all)
            g = sum(float(r["wait_ms"]) for r in ph_all) / n
            c = sum(float(r["cpu_ms"]) for r in ph_all) / n
            if g + c > 0:
                define(f"{mt}AmdahlCap", f"{(g + c) / max(c, 1e-9):.2f}")

        brca = [r for r in d.get("timings", []) if r["dataset"] == "brca2_150"]
        st = stats(brca, key=lambda r: r["label"] == "cpu_simd")
        if st:
            define(f"{mt}BrcaSimd", f"{st[0] * 1e3:.1f}")

        # hybrid split
        ph = [r for r in d.get("phases", [])
              if r["dataset"] == dataset and r["label"] == "hybrid_registers"]
        if ph:
            n = len(ph)
            gpu_ms = sum(float(r["wait_ms"]) for r in ph) / n
            cpu_ms = sum(float(r["cpu_ms"]) for r in ph) / n
            define(f"{mt}HybGpuMs", f"{gpu_ms:.0f}")
            define(f"{mt}HybCpuMs", f"{cpu_ms:.0f}")
            define(f"{mt}HybCpuPct", f"{100.0 * cpu_ms / max(gpu_ms + cpu_ms, 1e-9):.0f}")
            define(f"{mt}HybGpuLevels", str(int(float(ph[0]["gpu_levels"]))))
            define(f"{mt}HybCpuLevels", f"{int(float(ph[0]['cpu_levels'])):,}".replace(",", "\\,"))

        # The GPU's best on the dense levels against the CPU's best anywhere. Two peaks, which is
        # the honest way to say how much the GPU is worth where it has room to work.
        per_machine, _cells = kernel_split(levels_csv, {machine: d}, dataset)
        cpu_rates = [tot_cells / (st[0]) / 1e9
                     for lab in {r["label"] for r in rows if r["label"].startswith("cpu_")}
                     for st in [stats(rows, key=lambda r, l=lab: r["label"] == l)] if st]
        if per_machine.get(machine) and cpu_rates:
            best_wide = max(v[1] for v in per_machine[machine].values())
            worst_thin = min(v[2] for v in per_machine[machine].values())
            cpu_peak = max(cpu_rates)
            define(f"{mt}GpuDense", f"{best_wide:.0f}")
            define(f"{mt}GpuThin", f"{worst_thin:.2f}")
            define(f"{mt}CpuPeak", f"{cpu_peak:.1f}")
            define(f"{mt}DenseOverCpuPeak", f"{best_wide / cpu_peak:.0f}")
            define(f"{mt}DenseOverThin", f"{best_wide / max(worst_thin, 1e-9):.0f}")
            define(f"{mt}GpuTopThree", f"{max(v[0] for v in per_machine[machine].values()):.0f}")

        # The two reference points part two is read against.
        cb = cpu_baseline_gcups(data, machine, levels_csv, dataset)
        if cb:
            define(f"{mt}CpuLastColGcups", f"{cb:.1f}")
        if per_machine.get(machine, {}).get("gpu_last_col"):
            t3, dn, th = per_machine[machine]["gpu_last_col"]
            define(f"{mt}BaseLastColTopThree", f"{t3:.0f}")
            define(f"{mt}BaseLastColDense", f"{dn:.0f}")

        # per kernel, split at the scheduler's threshold: the ladder's own evidence
        acc = defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in d.get("levels", []):
            if r["dataset"] != dataset or r.get("mode") != "1":
                continue
            lvl = int(r["level"])
            if lvl not in level_width:
                continue
            w, b = level_width[lvl]
            slot = acc[r.get("label", "")]
            if w >= 16:
                slot[0] += b * QUERY_LEN.get(dataset, 149)
                slot[1] += float(r["ms"]) * 1e-3
            else:
                slot[2] += float(r["ms"]) * 1e-3

        ref = None
        for label in KERNEL_ORDER:
            a = acc.get(label)
            if not a or a[1] <= 0:
                continue
            gcs = a[0] / a[1] / 1e9
            if ref is None:
                ref = gcs
            short = tex_name(KERNEL_SHORT.get(label, label))
            define(f"{mt}Wide{short}", f"{gcs:.0f}")
            define(f"{mt}WideGain{short}", f"{gcs / ref:.2f}")
            define(f"{mt}Thin{short}", f"{a[2] * 1e3:.0f}")
            if per_machine.get(machine, {}).get(label):
                define(f"{mt}Top{short}", f"{per_machine[machine][label][0]:.0f}")

        # state of the art comparison, at the largest batch every engine reached
        hrows = [r for r in d.get("hga", []) if r["dataset"] == "brca2_150"]
        if hrows:
            by_engine = defaultdict(dict)
            for r in hrows:
                by_engine[r["engine"]][int(r["num_reads"])] = float(r["gcups"])
            common = set.intersection(*(set(v) for v in by_engine.values())) if by_engine else set()
            if common:
                at = max(common)
                define(f"{mt}HgaBatch", str(at))
                for engine, short in (("hga", "Hga"), ("gpu_multi", "GpuBatch"),
                                      ("cpu_multi", "CpuBatch")):
                    if engine in by_engine:
                        define(f"{mt}{short}Gcups", f"{by_engine[engine][at]:.1f}")
                if "hga" in by_engine and "gpu_multi" in by_engine:
                    define(f"{mt}VsHga",
                           f"{by_engine['gpu_multi'][at] / max(by_engine['hga'][at], 1e-9):.0f}")

            # HGA's own best case, from the batches it was given on its own: one read per thread.
            if "hga" in by_engine:
                # The smallest batch that reaches the plateau, not the largest measured: past
                # saturation the extra points differ by noise, and quoting the largest would
                # overstate how many reads HGA actually needs.
                top = max(by_engine["hga"].values())
                peak_at = min(r for r, g in by_engine["hga"].items() if g >= 0.99 * top)
                define(f"{mt}HgaPeakGcups", f"{by_engine['hga'][peak_at]:.1f}")
                define(f"{mt}HgaPeakBatch", f"{peak_at:,}".replace(",", "\\,"))
                if "gpu_multi" in by_engine and common:
                    best_ours = max(by_engine["gpu_multi"].values())
                    ours_at = max(by_engine["gpu_multi"],
                                  key=lambda r: by_engine["gpu_multi"][r])
                    define(f"{mt}OursPeakGcups", f"{best_ours:.1f}")
                    define(f"{mt}OursPeakBatch", f"{ours_at:,}".replace(",", "\\,"))
                    define(f"{mt}VsHgaPeak",
                           f"{best_ours / max(by_engine['hga'][peak_at], 1e-9):.1f}")
                    define(f"{mt}HgaBatchRatio", f"{peak_at // max(ours_at, 1)}")

        # batch saturation
        batch = [r for r in d.get("batch", []) if r["dataset"] == dataset]
        if batch:
            batch.sort(key=lambda r: int(r["num_reads"]))
            one = float(batch[0]["per_read_ms"])
            sat = float(batch[-1]["per_read_ms"])
            define(f"{mt}BatchMax", str(int(batch[-1]["num_reads"])))
            define(f"{mt}BatchPerRead", f"{sat:.1f}")
            define(f"{mt}BatchGain", f"{one / max(sat, 1e-9):.0f}")
            if best_ms:
                define(f"{mt}BatchVsHybrid", f"{best_ms / max(sat, 1e-9):.0f}")

        lines.append("")

    path = os.path.join(outdir, "numbers.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}  ({len(defined)} macros)")


# -------------------------------------------------------------------------------------------------

def save(fig, outdir, name):
    # No bbox_inches="tight" here: it re-crops to the ink and the PDF then comes out a few points
    # narrower than the figsize, so \includegraphics[width=\columnwidth] silently rescales it and
    # the fonts no longer match the document. tight_layout has already packed the axes inside the
    # requested size, so saving it verbatim is what keeps the figure at exactly one column.
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, dpi=220)
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--indir", default="outputs/report")
    ap.add_argument("-o", "--outdir", default="report")
    ap.add_argument("--dataset", default="150_10",
                    help="dataset the runtime and anatomy figures are about (default: 150_10)")
    ap.add_argument("--structure-datasets", nargs="+", default=["150_10", "brca2_150"],
                    help="the graphs the structure figure draws (default: one synthetic, one real)")
    ap.add_argument("--hga-dataset", default="brca2_150",
                    help="real graph the state-of-the-art comparison figure is about")
    ap.add_argument("--col-width", type=float, default=COL_W,
                    help=f"\\columnwidth in inches (default {COL_W}, measured from the report's "
                         "own geometry). Match this and includegraphics never rescales.")
    ap.add_argument("--text-width", type=float, default=TEXT_W,
                    help=f"\\textwidth in inches (default {TEXT_W})")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_all(args.indir)
    levels_csv = os.path.join(args.indir, "graph_levels.csv")

    if not data:
        print(f"no *_timings.csv in {args.indir}; run tools/report_bench.py first", file=sys.stderr)

    print(f"machines: {', '.join(sorted(data)) or '(none)'}")
    fig_structure(levels_csv, args.outdir, args.structure_datasets, args.col_width)
    fig_throughput(levels_csv, data, args.outdir, args.dataset, args.col_width)
    fig_kernels(levels_csv, data, args.outdir, args.dataset)
    fig_runtime(data, args.outdir, args.dataset, part="matrix",
                name="fig_runtime_matrix")
    fig_runtime(data, args.outdir, args.dataset, part="lastcol",
                name="fig_runtime_lastcol")
    table_hga(data, args.outdir, args.hga_dataset)
    write_numbers(data, levels_csv, args.outdir, args.dataset)


if __name__ == "__main__":
    main()
