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

  fig_anatomy    (a) where the hybrid's time goes
                 (b) per read time against batch size
                 -> why the remaining time is where it is, and what fixes it

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
    ax.text(0.985, 20, "16-node cut ", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=5.8, color=INK)

    # The figure carries no caption in the report, so the two regimes are named in the plot. Both
    # labels live in the empty band between the cut line and the legend, out of the data.
    pts = by_ds[drawn[0]]
    wide = [(l, w, b) for l, w, b in pts if w >= 16]
    total = sum(b for _, _, b in pts)
    if wide and total:
        pct = 100.0 * sum(b for _, _, b in wide) / total
        ax.annotate(f"{len(wide)} levels hold\n{pct:.0f}% of the work",
                    xy=(len(wide), 55), xytext=(34, 420),
                    fontsize=5.8, color=GROUP_COLOR["gpu"], ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=GROUP_COLOR["gpu"]))
        ax.annotate(f"{len(pts) - len(wide)} levels of 1-3 nodes",
                    xy=(900, 5.5), xytext=(30, 46),
                    fontsize=5.8, color=INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))

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
            ax.text(0.98, rate * 1.15, "CPU on the thin levels ", color=GROUP_COLOR["cpu"],
                    transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=5.8)

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
KERNEL_ORDER = ["gpu_shared_mem", "gpu_last_col", "gpu_warps", "gpu_registers",
                "gpu_merged_req"]

KERNEL_SHORT = {"gpu_shared_mem": "shared_mem", "gpu_last_col": "last_col",
                "gpu_warps": "warps", "gpu_registers": "registers",
                "gpu_merged_req": "merged_req"}


def fig_kernels(levels_csv, data, outdir, dataset, cut=16):
    """(a) throughput on the levels the hybrid keeps, (b) cost on the levels it gives away.

    Every rung is profiled on every level, then split at the scheduler's own threshold. The two
    panels move in opposite directions, which is the point of the whole report: each optimisation
    does buy what it was written to buy, on the levels it was written for, while making the thin
    levels worse. The hybrid exists to collect (a) without paying (b)."""
    widths = {}
    for r in read_csv(levels_csv):
        if r["dataset"] == dataset:
            widths[int(r["level"])] = (int(r["width"]), int(r["bases"]))
    if not widths:
        print(f"  no level widths for {dataset}, skipping fig_kernels", file=sys.stderr)
        return

    m = QUERY_LEN.get(dataset, 149)

    # machine -> label -> [wide cells, wide seconds, thin seconds]
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for machine, d in data.items():
        for r in d.get("levels", []):
            if r["dataset"] != dataset or r.get("mode") != "1":
                continue
            lvl = int(r["level"])
            if lvl not in widths:
                continue
            width, bases = widths[lvl]
            slot = acc[machine][r.get("label", f"v{r['version']}")]
            if width >= cut:
                slot[0] += bases * m
                slot[1] += float(r["ms"]) * 1e-3
            else:
                slot[2] += float(r["ms"]) * 1e-3

    machines = [mc for mc in sorted(acc) if acc[mc]]
    if not machines:
        print("  no per level data, skipping fig_kernels", file=sys.stderr)
        return

    labels = [l for l in KERNEL_ORDER if any(l in acc[mc] for mc in machines)]
    labels += sorted({l for mc in machines for l in acc[mc]} - set(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 0.30 * len(labels) + 1.05))
    y = list(range(len(labels)))
    h = 0.72 / max(len(machines), 1)

    for k, machine in enumerate(machines):
        off = (k - (len(machines) - 1) / 2) * h
        color = GROUP_COLOR["hybrid"] if machine == "spark" else GROUP_COLOR["gpu"]

        wide = [acc[machine][l][0] / acc[machine][l][1] / 1e9 if acc[machine].get(l)
                and acc[machine][l][1] > 0 else 0.0 for l in labels]
        thin = [acc[machine][l][2] * 1e3 if acc[machine].get(l) else 0.0 for l in labels]

        ax1.barh([v + off for v in y], wide, height=h * 0.92, color=color, zorder=3,
                 label=machine)
        ax2.barh([v + off for v in y], thin, height=h * 0.92, color=GROUP_COLOR["cpu"],
                 zorder=3, label=machine)

        base = wide[0] if wide and wide[0] > 0 else None
        for i, v in enumerate(wide):
            if v <= 0:
                continue
            tag = f"{v:.0f}" + (f"  ({v / base:.2f}x)" if base else "")
            ax1.text(v * 1.02, i + off, tag, va="center", ha="left", fontsize=6.0, color=INK)
        for i, v in enumerate(thin):
            if v > 0:
                ax2.text(v * 1.02, i + off, f"{v:.0f}", va="center", ha="left",
                         fontsize=6.0, color=INK)

    for ax, xlab, title in (
            (ax1, f"Gcell/s on levels of $\\geq${cut} nodes",
             f"(a) what the GPU achieves where the hybrid uses it"),
            (ax2, f"ms on levels of $<${cut} nodes",
             f"(b) what the same kernel costs where it does not")):
        ax.set_yticks(y)
        ax.set_yticklabels([KERNEL_SHORT.get(l, l) for l in labels], color=INK2)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left", color=INK)
        clean_ax(ax)

    ax1.set_xlim(0, max([acc[mc][l][0] / acc[mc][l][1] / 1e9
                         for mc in machines for l in labels
                         if acc[mc].get(l) and acc[mc][l][1] > 0] or [1]) * 1.35)
    ax2.set_xlim(0, max([acc[mc][l][2] * 1e3 for mc in machines for l in labels
                         if acc[mc].get(l)] or [1]) * 1.20)

    if len(machines) > 1:
        ax1.legend(frameon=False, loc="lower right", labelcolor=INK2)

    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_kernels")


# -------------------------------------------------------------------------------------------------
#                        Figure 2: the result
# -------------------------------------------------------------------------------------------------

def fig_runtime(data, outdir, dataset):
    """Wall time per implementation, one panel per machine, in the report's narrative order."""
    machines = [m for m in sorted(data) if data[m].get("timings")]
    if not machines:
        print("  no timings to draw", file=sys.stderr)
        return

    # A single fixed ordering across panels, taken from the CSV so it follows report_bench's PLAN.
    order, seen = [], set()
    for m in machines:
        for r in data[m]["timings"]:
            key = (r["label"], r["group"])
            if key not in seen:
                seen.add(key)
                order.append(key)

    n = len(machines)
    fig, axes = plt.subplots(1, n, figsize=(TEXT_W if n > 1 else COL_W, 0.142 * len(order) + 0.88),
                             squeeze=False, sharey=True)

    for ax, machine in zip(axes[0], machines):
        rows = [r for r in data[machine]["timings"] if r["dataset"] == dataset]
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
        for i, v in enumerate(values):
            ax.text(v * 1.10, i, fmt_ms(v), va="center", ha="left", fontsize=6.2,
                    color=INK if v > best * 1.02 else GROUP_COLOR["hybrid"])

        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=INK2)
        ax.set_ylim(len(values) - 0.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlim(best * 0.55, max(values) * 3.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("wall time per alignment (ms, log)")
        ax.set_title(machine_title(machine, data[machine].get("info")), loc="left", color=INK)
        clean_ax(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g])
               for g in ("cpu", "gpu", "fair", "hybrid") if any(g == gr for _, gr in order)]
    names = [GROUP_NAME[g] for g in ("cpu", "gpu", "fair", "hybrid")
             if any(g == gr for _, gr in order)]
    fig.legend(handles, names, frameon=False, ncol=len(names), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK2)

    fig.tight_layout(pad=0.4, rect=(0, 0.055, 1, 1))
    save(fig, outdir, "fig_runtime")


# -------------------------------------------------------------------------------------------------
#                        Figure 3: why, and what fixes it
# -------------------------------------------------------------------------------------------------

def fig_anatomy(data, outdir, dataset):
    """(a) where the hybrid's time goes, (b) per read time against batch size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.80),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- (a) hybrid phase breakdown --------------------------------------------------------------
    bars, labels = [], []
    for machine in sorted(data):
        rows = [r for r in data[machine].get("phases", [])
                if r["dataset"] == dataset and r["label"] == "hybrid_registers"]
        if not rows:
            rows = [r for r in data[machine].get("phases", []) if r["dataset"] == dataset]
        if not rows:
            continue
        n = len(rows)
        bars.append((sum(float(r["wait_ms"]) for r in rows) / n,
                     sum(float(r["cpu_ms"]) for r in rows) / n,
                     sum(float(r["traceback_ms"]) + float(r["launch_ms"]) for r in rows) / n))
        labels.append(machine)

    if bars:
        y = list(range(len(bars)))
        gpu = [b[0] for b in bars]
        cpu = [b[1] for b in bars]
        rest = [b[2] for b in bars]

        ax1.barh(y, gpu, height=0.6, color=GROUP_COLOR["gpu"], zorder=3, label="GPU head")
        ax1.barh(y, cpu, left=gpu, height=0.6, color=GROUP_COLOR["cpu"], zorder=3, label="CPU tail")
        ax1.barh(y, rest, left=[g + c for g, c in zip(gpu, cpu)], height=0.6,
                 color=INK2, zorder=3, label="rest")

        for i, (g, c, r) in enumerate(bars):
            ax1.text(g / 2, i, f"{g:.0f}", va="center", ha="center", fontsize=6.2, color="white")
            ax1.text(g + c / 2, i, f"{c:.0f}", va="center", ha="center", fontsize=6.2, color="white")
            ax1.text(g + c + r, i - 0.42, f"{100 * c / (g + c + r):.0f}% on 14% of the work",
                     va="bottom", ha="right", fontsize=5.8, color=INK2)

        ax1.set_yticks(y)
        ax1.set_yticklabels(labels, color=INK2)
        ax1.set_ylim(len(bars) - 0.5, -0.5)
        ax1.set_xlim(0, max(sum(b) for b in bars) * 1.04)
        ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                   labelcolor=INK2, handlelength=0.8, borderpad=0.0, columnspacing=0.9,
                   handletextpad=0.35)
    else:
        ax1.text(0.5, 0.5, "no phase data", transform=ax1.transAxes, ha="center", color=INK2)

    ax1.set_xlabel("ms per alignment")
    ax1.set_title("(a) the tail the GPU\nrefuses dominates", loc="left", color=INK)
    clean_ax(ax1)

    # ---- (b) per read time against batch size ----------------------------------------------------
    drew = False
    for machine in sorted(data):
        rows = [r for r in data[machine].get("batch", []) if r["dataset"] == dataset]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["num_reads"]))
        ax2.plot([int(r["num_reads"]) for r in rows], [float(r["per_read_ms"]) for r in rows],
                 "o-", ms=3.2, lw=1.2, color=GROUP_COLOR["hybrid"], label=machine, zorder=3)
        drew = True

    # The single read hybrid, as the line a batch has to beat.
    for machine in sorted(data):
        s = stats([r for r in data[machine].get("timings", [])
                   if r["dataset"] == dataset and r["label"] == "hybrid_registers"])
        if s:
            ax2.axhline(s[0] * 1e3, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax2.text(0.98, s[0] * 1e3, "hybrid, 1 read ", transform=ax2.get_yaxis_transform(),
                     va="bottom", ha="right", fontsize=6.0, color=INK2)
            break

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    # A log axis over less than two decades gets one labelled tick from the default locator, which
    # is not enough to read a curve off, so the ticks are placed by hand over the observed range.
    if drew:
        lo = min(float(r["per_read_ms"]) for m in data for r in data[m].get("batch", [])
                 if r["dataset"] == dataset)
        hi = max(float(r["per_read_ms"]) for m in data for r in data[m].get("batch", [])
                 if r["dataset"] == dataset)
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo * 0.8 <= t <= hi * 1.25]
        if len(ticks) >= 2:
            ax2.set_yticks(ticks)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax2.set_xlabel("reads per batch")
    ax2.set_ylabel("ms per read")
    ax2.set_title("(b) batching pays\nmore than tuning", loc="left", color=INK)
    if drew:
        ax2.legend(frameon=False, loc="upper right", labelcolor=INK2)
    else:
        ax2.text(0.5, 0.5, "no batch data", transform=ax2.transAxes, ha="center", color=INK2)
    clean_ax(ax2, xgrid=False, ygrid=True)

    fig.tight_layout(pad=0.35)
    save(fig, outdir, "fig_anatomy")


# -------------------------------------------------------------------------------------------------
#                        Figure: against the state of the art
# -------------------------------------------------------------------------------------------------

ENGINE_STYLE = {
    "hga":       ("HGA (ICPP'21)", INK2,                 "s", (0, (3, 2))),
    "gpu_multi": ("ours, GPU batch", GROUP_COLOR["hybrid"], "o", "-"),
    "cpu_multi": ("ours, CPU batch", GROUP_COLOR["cpu"],    "^", "-"),
}


def fig_hga(data, outdir, dataset, width=COL_W):
    """Throughput against batch size for the three engines, on the real graph.

    GCUPS is the unit HGA's paper reports and is the same quantity as our Gcell/s, because their
    one-vertex-per-base graph holds exactly the bases our nodes hold. Batch size is the x axis
    because that is the only axis any of the three can scale on: a real pangenome graph is a thin
    chain, so there is no node parallelism to exploit within one read."""
    machines = [m for m in sorted(data)
                if any(r["dataset"] == dataset for r in data[m].get("hga", []))]
    if not machines:
        print(f"  no hga rows for {dataset}, skipping fig_hga", file=sys.stderr)
        return

    n = len(machines)
    fig, axes = plt.subplots(1, n, figsize=(TEXT_W if n > 1 else width, width * 0.62),
                             squeeze=False, sharey=True)

    for ax, machine in zip(axes[0], machines):
        rows = [r for r in data[machine]["hga"] if r["dataset"] == dataset]
        for engine, (label, color, marker, ls) in ENGINE_STYLE.items():
            pts = sorted((int(r["num_reads"]), float(r["gcups"]))
                         for r in rows if r["engine"] == engine)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, ls=ls, ms=3.0,
                    lw=1.2, color=color, label=label, zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("reads per batch")
        ax.set_title(machine_title(machine, data[machine].get("info")), loc="left", color=INK)
        clean_ax(ax, xgrid=False, ygrid=True)

    axes[0][0].set_ylabel("GCUPS")
    axes[0][0].legend(frameon=False, loc="upper left", labelcolor=INK2, handlelength=1.6,
                      borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    save(fig, outdir, "fig_hga")


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
    lines = ["% Generated by tools/report_figs.py - do not edit, re-run the script.",
             "% \\input{numbers} in the preamble, then use e.g. \\SparkHybridRegisters.",
             ""]
    defined = set()

    def define(name, value):
        if name in defined:
            print(f"  name clash on \\{name}, skipped (fix the tag map)", file=sys.stderr)
            return
        defined.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---- graph structure, machine independent ---------------------------------------------------
    rows = read_csv(levels_csv)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append((int(r["width"]), int(r["bases"])))

    level_width = {int(r["level"]): (int(r["width"]), int(r["bases"]))
                   for r in rows if r["dataset"] == dataset}

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

    lines.append("")

    # ---- per machine ----------------------------------------------------------------------------
    for machine, d in sorted(data.items()):
        mt = machine_tag(machine)
        info = d.get("info", {})
        define(f"{mt}Gpu", info.get("gpu", "?").replace("NVIDIA ", ""))
        define(f"{mt}Cpu", info.get("cpu", "?").split("@")[0].strip())
        define(f"{mt}Cores", str(info.get("cores", "?")))

        rows = [r for r in d.get("timings", []) if r["dataset"] == dataset]
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
    print(f"  wrote {path}  ({sum(1 for l in lines if l.startswith(chr(92) + 'newcommand'))} macros)")


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
    fig_runtime(data, args.outdir, args.dataset)
    fig_anatomy(data, args.outdir, args.dataset)
    fig_hga(data, args.outdir, args.hga_dataset)
    write_numbers(data, levels_csv, args.outdir, args.dataset)


if __name__ == "__main__":
    main()
